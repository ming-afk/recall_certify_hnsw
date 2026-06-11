from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def import_hnswlib_with_optional_override():
    """Import hnswlib, optionally forcing a specific extension path via env."""

    override = os.environ.get("HNSWLIB_SO_OVERRIDE", "").strip()
    if not override:
        import hnswlib  # type: ignore

        return hnswlib

    so_path = Path(override)
    if not so_path.exists():
        raise FileNotFoundError(f"HNSWLIB_SO_OVERRIDE does not exist: {so_path}")

    # Ensure we load the requested extension file for module name "hnswlib".
    sys.modules.pop("hnswlib", None)
    spec = importlib.util.spec_from_file_location("hnswlib", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load hnswlib from override path: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["hnswlib"] = mod
    return mod


def infer_dataset_tag(index_path: str, run_args: Optional[Dict[str, Any]] = None) -> str:
    """Infer dataset key used by cache naming."""

    low = str(index_path).lower()
    if "sift" in low:
        return "sift"
    if "gist" in low:
        return "gist"
    if "deep1m" in low or "deep10m" in low or "deep" in low:
        return "deep1m"

    if run_args is not None:
        for key in ("dataset_file", "deep_base_file", "fvec_base_file", "index_path"):
            v = str(run_args.get(key, "")).lower()
            if "sift" in v:
                return "sift"
            if "gist" in v:
                return "gist"
            if "deep" in v:
                return "deep1m"
    raise ValueError(f"Could not infer dataset from index path: {index_path}")


def infer_m_from_index_path(index_path: str) -> Optional[int]:
    m = re.search(r"_m(\d+)_", str(index_path).lower())
    if m:
        return int(m.group(1))
    m2 = re.search(r"m(\d+)", Path(index_path).stem.lower())
    if m2:
        return int(m2.group(1))
    return None


def default_pca_dims_for_dataset(dataset_tag: str) -> int:
    # Matches the tested setup:
    #   SIFT: 32, DEEP1M: 32, GIST: 64
    if dataset_tag == "gist":
        return 64
    return 32


def default_pca_cache_path(dataset_tag: str, pca_dims: int) -> Path:
    base = Path(__file__).resolve().parent / "pca_cache"
    candidates = []
    if dataset_tag == "gist" and pca_dims == 64:
        candidates.append(base / "gist_M16_efc100_m64_from120_pca.npz")
    candidates.append(base / f"{dataset_tag}_M16_efc100_m{int(pca_dims)}_pca.npz")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No PCA cache found for dataset={dataset_tag}, dims={pca_dims}. Tried: {candidates}"
    )


def default_sparse_map_path(dataset_tag: str, index_path: str, l1_ef: int, map_dir: Path) -> Path:
    stem = Path(index_path).stem
    return map_dir / f"{dataset_tag}_{stem}_sparse_l1_k1_l1ef{int(l1_ef)}.npz"


def load_pca_into_index(index: Any, pca_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(pca_path)
    if "pca_dirs" not in z or "projX" not in z:
        raise ValueError(f"Invalid PCA cache: {pca_path} (missing pca_dirs/projX)")
    pca_dirs = np.asarray(z["pca_dirs"], dtype=np.float32)
    proj_x = np.asarray(z["projX"], dtype=np.float32)
    index.set_pca_data(pca_dirs, proj_x)
    return pca_dirs, proj_x


def ensure_sparse_lookup(index: Any, map_path: Path, precompute_k_trace: int = 1) -> str:
    """Load sparse L1 map from disk when available, else build and save."""

    if map_path.exists():
        z = np.load(map_path)
        if "l0_labels" not in z or "parent_l1_labels" not in z:
            raise ValueError(f"Invalid sparse map file: {map_path}")
        index.load_sparse_l1_lookup(z["l0_labels"], z["parent_l1_labels"])
        return "loaded"

    index.build_sparse_l1_lookup(k_trace=int(precompute_k_trace))
    lookup = index.get_sparse_l1_lookup()
    l0_labels = np.asarray(lookup["l0_labels"], dtype=np.int64)
    parent_l1_labels = np.asarray(lookup["parent_l1_labels"], dtype=np.int64)

    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(map_path, l0_labels=l0_labels, parent_l1_labels=parent_l1_labels)
    return "built"


@dataclass
class SparseRectifyConfig:
    k: int
    l1_k: int
    l1_t: float
    l1_ef: int
    use_pca: bool
    subtract_ann_time: bool


@dataclass
class MBVRectifyConfig:
    k: int
    t: float
    seed_count: int
    seed_start: int
    subtract_ann_time: bool


def run_sparse_rectify_query(
    index: Any,
    q: np.ndarray,
    gt_k: np.ndarray,
    cfg: SparseRectifyConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]:
    labels, dists, stats = index.sparse_l1_full_query_ex(
        np.asarray(q, dtype=np.float32),
        int(cfg.k),
        k_l1=int(cfg.l1_k),
        t_l1=float(cfg.l1_t),
        ef_l1=int(cfg.l1_ef),
        use_pca=bool(cfg.use_pca),
        euclidean=True,
        gt_labels=np.asarray(gt_k, dtype=np.int64),
    )

    total_ns = int(stats.get("total_time_ns", 0))
    if cfg.subtract_ann_time:
        total_ns -= int(stats.get("ann_time_ns", 0))
    if total_ns < 0:
        total_ns = 0

    return np.asarray(labels, dtype=np.int64), np.asarray(dists), stats, total_ns


def run_mbv_rectify_query(
    index: Any,
    q: np.ndarray,
    gt_k: np.ndarray,
    cfg: MBVRectifyConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]:
    labels, dists, stats = index.mbv_query_ex(
        np.asarray(q, dtype=np.float32),
        int(cfg.k),
        t=float(cfg.t),
        euclidean=True,
        use_ann_cache=True,
        use_ellipse_prune=True,
        collect_node_trace=False,
        oracle_stop_pop=-1,
        mbv_seed_count=int(cfg.seed_count),
        mbv_seed_start=int(cfg.seed_start),
    )

    total_ns = int(stats.get("total_time_ns", 0))
    if cfg.subtract_ann_time:
        total_ns -= int(stats.get("ann_time_ns", 0))
    if total_ns < 0:
        total_ns = 0

    return np.asarray(labels, dtype=np.int64), np.asarray(dists), stats, total_ns
