#!/usr/bin/env python3
from __future__ import annotations

"""One-shot CRC certifier training/evaluation runner.

This script is the reproducible entry point used by downstream alpha/tau sweep
scripts. It trains the certifier on a calibration split, evaluates on held-out
queries, and writes all artifacts needed to replay later experiments.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import crc_core

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

try:
    import hnswlib  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("hnswlib is required to run this script") from e


# -----------------------------
# Dataset loading helpers
# -----------------------------

def _read_ivecs(path: Path) -> np.ndarray:
    """Read Faiss/ANN-benchmarks .ivecs file into int matrix (N, D)."""

    arr = np.fromfile(path, dtype=np.int32)
    if arr.size == 0:
        raise ValueError(f"Empty ivecs file: {path}")
    dim = int(arr[0])
    if dim <= 0:
        raise ValueError(f"Invalid ivecs width in {path}: {dim}")
    row = dim + 1
    if arr.size % row != 0:
        raise ValueError(f"Malformed ivecs file: {path}")
    return arr.reshape(-1, row)[:, 1:].copy()


def _read_fvecs(path: Path) -> np.ndarray:
    """Read Faiss .fvecs file into float32 matrix (N, D)."""

    arr = np.fromfile(path, dtype=np.int32)
    if arr.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    dim = int(arr[0])
    if dim <= 0:
        raise ValueError(f"Invalid fvecs width in {path}: {dim}")
    row = dim + 1
    if arr.size % row != 0:
        raise ValueError(f"Malformed fvecs file: {path}")
    arr = arr.reshape(-1, row)
    return np.asarray(arr[:, 1:].view(np.float32), dtype=np.float32)


def _read_fbin(path: Path, dim: Optional[int] = None, dtype=np.float32) -> np.ndarray:
    """Read raw or headered FBIN vectors with known embedding dimension."""

    raw = np.fromfile(path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"Empty binary vector file: {path}")
    if dim is None:
        raise ValueError(f"dim must be provided to read fbin file: {path}")
    d = int(dim)
    if raw.size % d == 0:
        return raw.reshape(-1, d)

    # Fallback: support headered FBIN format: int32(n), int32(d), then n*d float32 payload.
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Malformed fbin header: {path}")
        n_h, d_h = int(header[0]), int(header[1])
        payload = np.fromfile(f, dtype=dtype)
    if d_h != d:
        raise ValueError(f"fbin header dim={d_h} does not match expected dim={d}: {path}")
    if payload.size != n_h * d_h:
        raise ValueError(f"Malformed fbin payload in {path}: got {payload.size}, expected {n_h * d_h}")
    return payload.reshape(n_h, d_h)


def _load_ann_hdf5(dataset_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ANN-benchmarks-like HDF5 dataset (base/query/ground-truth)."""

    if h5py is None:
        raise SystemExit("h5py is required for .hdf5 datasets")
    with h5py.File(dataset_file, "r") as f:
        base_key = None
        query_key = None
        gt_key = None
        for cand in ("train", "base"):
            if cand in f:
                base_key = cand
                break
        for cand in ("test", "query", "queries"):
            if cand in f:
                query_key = cand
                break
        for cand in ("neighbors", "gt", "groundtruth"):
            if cand in f:
                gt_key = cand
                break
        if base_key is None or query_key is None or gt_key is None:
            raise ValueError(
                f"Could not find required datasets in {dataset_file}. "
                "Need base/train, test/query, and neighbors/gt."
            )
        x_train = np.asarray(f[base_key], dtype=np.float32)
        x_query = np.asarray(f[query_key], dtype=np.float32)
        gt = np.asarray(f[gt_key], dtype=np.int64)
    return x_train, x_query, gt


def _load_deep1m(base_file: Path, query_file: Path, gt_file: Path, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Deep1M-style FBIN+IVECS dataset triplet."""

    x_train = np.asarray(_read_fbin(base_file, dim=dim), dtype=np.float32)
    x_query = np.asarray(_read_fbin(query_file, dim=dim), dtype=np.float32)
    gt = np.asarray(_read_ivecs(gt_file), dtype=np.int64)
    return x_train, x_query, gt


def _translate_gt_to_dense(
    gt_raw: np.ndarray,
    base_ids: np.ndarray,
    gt_label_offset: int = 0,
) -> np.ndarray:
    """Translate GT labels into dense ids after optional base permutation."""

    base_to_dense = {int(b): int(i) for i, b in enumerate(base_ids.tolist())}
    if len(base_to_dense) != int(base_ids.shape[0]):
        raise ValueError("base_ids must be unique for gt translation")

    def _map_single(v: int, offset: int) -> int:
        key = int(v - offset)
        if key in base_to_dense:
            return int(base_to_dense[key])
        raise KeyError

    out = np.empty_like(gt_raw, dtype=np.int64)
    fallback_used = False
    for i in range(gt_raw.shape[0]):
        for j in range(gt_raw.shape[1]):
            g = int(gt_raw[i, j])
            try:
                out[i, j] = _map_single(g, int(gt_label_offset))
            except KeyError:
                try:
                    out[i, j] = _map_single(g, 1)
                    fallback_used = True
                except KeyError as e:
                    raise ValueError(
                        f"GT label {g} (row={i}, col={j}) could not be translated to dense ids"
                    ) from e
    if fallback_used:
        print("GT translation fallback applied: subtracting 1 from GT labels.")
    return out


def _load_fvecs_ivecs(
    base_file: Path,
    query_file: Path,
    gt_file: Path,
    base_permute_seed: Optional[int] = None,
    gt_label_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load fvecs/ivecs triplet, optionally permuting base with dense-id GT remap."""

    x_base_full = np.asarray(_read_fvecs(base_file), dtype=np.float32)
    x_query = np.asarray(_read_fvecs(query_file), dtype=np.float32)
    gt_raw = np.asarray(_read_ivecs(gt_file), dtype=np.int64)

    if base_permute_seed is None:
        x_train = x_base_full
        gt = gt_raw
    else:
        n = int(x_base_full.shape[0])
        rng = np.random.default_rng(int(base_permute_seed))
        base_ids = rng.choice(n, size=n, replace=False).astype(np.int64)
        x_train = np.asarray(x_base_full[base_ids], dtype=np.float32)
        gt = _translate_gt_to_dense(gt_raw, base_ids, gt_label_offset=gt_label_offset)

    return x_train, x_query, gt


# -----------------------------
# Index helpers
# -----------------------------

def dataset_tag_from_args(args: argparse.Namespace) -> str:
    """Build a stable dataset tag used in automatic index path naming."""

    if args.dataset_file is not None:
        return args.dataset_file.stem
    if args.deep_base_file is not None:
        return args.deep_base_file.stem.replace("_base", "")
    if args.fvec_base_file is not None:
        return args.fvec_base_file.stem.replace("_base", "")
    return "dataset"


def resolve_index_path(index_dir: Path, dataset_tag: str, M: int, efc: int) -> Path:
    """Compute default serialized index path."""

    return index_dir / f"{dataset_tag}_M{int(M)}_efc{int(efc)}.bin"


def ensure_index(x_train: np.ndarray, index_path: Path, M: int, efc: int, rebuild_index: bool, space: str) -> hnswlib.Index:
    """Load existing HNSW index or build/save a new one."""

    dim = int(x_train.shape[1])
    idx = hnswlib.Index(space=space, dim=dim)
    if index_path.exists() and not rebuild_index:
        idx.load_index(str(index_path), max_elements=int(x_train.shape[0]))
        return idx

    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx.init_index(max_elements=int(x_train.shape[0]), M=int(M), ef_construction=int(efc))
    idx.add_items(np.asarray(x_train, dtype=np.float32), ids=np.arange(x_train.shape[0], dtype=np.int64))
    idx.save_index(str(index_path))
    return idx


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for all supported dataset modes."""

    p = argparse.ArgumentParser(
        description="Run CRC calibration/evaluation in the same style as your existing HNSW experiment scripts."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-file", type=Path, help="ANN-benchmarks style .hdf5 dataset, e.g. mnist-784-euclidean.hdf5")
    src.add_argument("--deep-base-file", type=Path, help="Deep1M base .fbin file")
    src.add_argument("--fvec-base-file", type=Path, help="Base .fvecs file (e.g., sift_base.fvecs or gist_base.fvecs)")

    p.add_argument("--deep-query-file", type=Path, default=None, help="Deep1M query .fbin file")
    p.add_argument("--deep-gt-file", type=Path, default=None, help="Deep1M ground-truth .ivecs file")
    p.add_argument("--fvec-query-file", type=Path, default=None, help="Query .fvecs file")
    p.add_argument("--fvec-gt-file", type=Path, default=None, help="Ground-truth .ivecs file")
    p.add_argument("--fvec-base-permute-seed", type=int, default=None, help="Optional base permutation seed for dense-id indexing (useful for GIST workflows)")
    p.add_argument("--gt-label-offset", type=int, default=0, help="Optional offset subtracted from GT ids before translation")
    p.add_argument("--dim", type=int, default=None, help="Required for fbin datasets; optional otherwise")

    p.add_argument("--index-dir", type=Path, default=Path("indexes"))
    p.add_argument("--index-path", type=Path, default=None, help="Optional explicit serialized index path; overrides auto naming")
    p.add_argument("--rebuild-index", action="store_true")

    p.add_argument("--space", type=str, default="l2", choices=["l2", "ip", "cosine"])
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--efc", type=int, default=100)
    p.add_argument("--efs", type=int, default=40)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.95)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cal-size", type=int, default=1000)
    p.add_argument("--qcnt", type=int, default=None, help="Use only the first qcnt queries after shuffling/splitting")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-augmented", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_dataset_from_args(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch dataset loading based on mutually-exclusive source flags."""

    if args.dataset_file is not None:
        return _load_ann_hdf5(args.dataset_file)

    if args.deep_base_file is not None:
        if args.deep_query_file is None or args.deep_gt_file is None:
            raise ValueError("For Deep1M mode you must provide --deep-base-file, --deep-query-file, and --deep-gt-file")
        if args.dim is None:
            raise ValueError("For Deep1M .fbin input you must provide --dim")
        return _load_deep1m(args.deep_base_file, args.deep_query_file, args.deep_gt_file, args.dim)

    if args.fvec_base_file is not None:
        if args.fvec_query_file is None or args.fvec_gt_file is None:
            raise ValueError("For fvecs mode you must provide --fvec-base-file, --fvec-query-file, and --fvec-gt-file")
        return _load_fvecs_ivecs(
            args.fvec_base_file,
            args.fvec_query_file,
            args.fvec_gt_file,
            base_permute_seed=args.fvec_base_permute_seed,
            gt_label_offset=args.gt_label_offset,
        )

    raise ValueError("No dataset source provided")


def main() -> None:
    """Train CRC certifier, evaluate, and write reproducible output artifacts."""

    args = parse_args()

    x_train, x_query, gt = load_dataset_from_args(args)
    if x_query.shape[0] != gt.shape[0]:
        raise ValueError("query count and gt row count must match")
    if gt.shape[1] < args.k:
        raise ValueError("ground truth must have at least k columns")
    if args.cal_size >= x_query.shape[0]:
        raise ValueError("cal-size must be smaller than the number of queries")
    if args.qcnt is not None and args.qcnt <= 1:
        raise ValueError("qcnt must be > 1 when provided")

    dataset_tag = dataset_tag_from_args(args)
    index_path = args.index_path or resolve_index_path(args.index_dir, dataset_tag, args.M, args.efc)
    idx = ensure_index(x_train, index_path, args.M, args.efc, args.rebuild_index, args.space)
    idx.set_ef(int(args.efs))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(x_query.shape[0])
    if args.qcnt is not None:
        perm = perm[: min(int(args.qcnt), perm.size)]

    cal_n = min(int(args.cal_size), perm.size - 1)
    cal_idx = perm[:cal_n]
    test_idx = perm[cal_n:]
    if test_idx.size == 0:
        raise ValueError("No test queries left after calibration split")

    X_cal = np.asarray(x_query[cal_idx], dtype=np.float32)
    GT_cal = np.asarray(gt[cal_idx, : args.k], dtype=np.int64)
    X_test = np.asarray(x_query[test_idx], dtype=np.float32)
    GT_test = np.asarray(gt[test_idx, : args.k], dtype=np.int64)

    certifier, cal_rows = crc_core.fit_crc_certifier(
        index=idx,
        X_cal=X_cal,
        GT_cal=GT_cal,
        k=args.k,
        ef_search=args.efs,
        tau=args.tau,
        alpha=args.alpha,
        use_augmented=args.use_augmented,
    )
    metrics = crc_core.evaluate_crc(idx, X_test, GT_test, certifier)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crc_core.save_certifier(certifier, args.out_dir / "certifier.json")
    crc_core.save_calibration_rows(cal_rows, args.out_dir / "calibration_rows.csv")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run_args = {
        **vars(args),
        "dataset_tag": dataset_tag,
        "index_path": str(index_path),
        "n_base": int(x_train.shape[0]),
        "n_query": int(x_query.shape[0]),
        "dim_resolved": int(x_train.shape[1]),
    }
    (args.out_dir / "run_args.json").write_text(json.dumps(run_args, indent=2, default=str))

    print(json.dumps({"certifier": certifier.to_dict(), "metrics": metrics, "index_path": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
