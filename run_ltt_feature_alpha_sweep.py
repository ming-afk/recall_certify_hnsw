#!/usr/bin/env python3
from __future__ import annotations

"""LTT alpha sweep with reusable feature/model combinations.

This script shares split/index metadata with CRC runs through `run_args.json`
so side-by-side comparisons remain reproducible.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py  # type: ignore
import numpy as np
from scipy.stats import binom  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

import crc_core
import sparse_rectify as sparse


def _read_ivecs(path: Path) -> np.ndarray:
    """Read .ivecs file into int matrix (N, D)."""

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
    """Read .fvecs file into float32 matrix (N, D)."""

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


def _read_fbin(path: Path, dim: int) -> np.ndarray:
    """Read raw or headered FBIN vectors with known dimension."""

    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty fbin file: {path}")
    d = int(dim)
    if raw.size % d == 0:
        return raw.reshape(-1, d)

    # Fallback: support headered FBIN format: int32(n), int32(d), then n*d float32 payload.
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Malformed fbin header: {path}")
        n_h, d_h = int(header[0]), int(header[1])
        payload = np.fromfile(f, dtype=np.float32)
    if d_h != d:
        raise ValueError(f"fbin header dim={d_h} does not match expected dim={d}: {path}")
    if payload.size != n_h * d_h:
        raise ValueError(f"Malformed fbin payload in {path}: got {payload.size}, expected {n_h * d_h}")
    return np.asarray(payload.reshape(n_h, d_h), dtype=np.float32)


def _translate_gt_to_dense(gt_raw: np.ndarray, base_ids: np.ndarray, gt_label_offset: int = 0) -> np.ndarray:
    """Translate GT ids to dense ids after base permutation."""

    base_to_dense = {int(b): int(i) for i, b in enumerate(base_ids.tolist())}
    if len(base_to_dense) != int(base_ids.shape[0]):
        raise ValueError("base_ids must be unique for gt translation")

    def _map_single(v: int, offset: int) -> int:
        key = int(v - offset)
        if key in base_to_dense:
            return int(base_to_dense[key])
        raise KeyError

    out = np.empty_like(gt_raw, dtype=np.int64)
    for i in range(gt_raw.shape[0]):
        for j in range(gt_raw.shape[1]):
            g = int(gt_raw[i, j])
            try:
                out[i, j] = _map_single(g, int(gt_label_offset))
            except KeyError:
                try:
                    out[i, j] = _map_single(g, 1)
                except KeyError as e:
                    raise ValueError(f"GT label {g} (row={i}, col={j}) could not be translated to dense ids") from e
    return out


def _load_ann_hdf5(dataset_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load base/query/ground-truth arrays from ANN-style HDF5."""

    with h5py.File(dataset_file, "r") as f:
        base_key = "train" if "train" in f else "base"
        query_key = "test" if "test" in f else ("query" if "query" in f else "queries")
        gt_key = "neighbors" if "neighbors" in f else ("gt" if "gt" in f else "groundtruth")
        x_train = np.asarray(f[base_key], dtype=np.float32)
        x_query = np.asarray(f[query_key], dtype=np.float32)
        gt = np.asarray(f[gt_key], dtype=np.int64)
    return x_train, x_query, gt


def _load_dataset_from_run_args(run_args: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from the source fields saved in run_args.json."""

    dataset_file = run_args.get("dataset_file")
    if dataset_file:
        return _load_ann_hdf5(Path(str(dataset_file)))

    deep_base = run_args.get("deep_base_file")
    if deep_base:
        deep_query = run_args.get("deep_query_file")
        deep_gt = run_args.get("deep_gt_file")
        dim = run_args.get("dim")
        if not deep_query or not deep_gt or dim is None:
            raise ValueError("run_args deep mode missing deep_query_file/deep_gt_file/dim")
        x_train = np.asarray(_read_fbin(Path(str(deep_base)), int(dim)), dtype=np.float32)
        x_query = np.asarray(_read_fbin(Path(str(deep_query)), int(dim)), dtype=np.float32)
        gt = np.asarray(_read_ivecs(Path(str(deep_gt))), dtype=np.int64)
        return x_train, x_query, gt

    fvec_base = run_args.get("fvec_base_file")
    if fvec_base:
        fvec_query = run_args.get("fvec_query_file")
        fvec_gt = run_args.get("fvec_gt_file")
        if not fvec_query or not fvec_gt:
            raise ValueError("run_args fvec mode missing fvec_query_file/fvec_gt_file")
        x_base_full = np.asarray(_read_fvecs(Path(str(fvec_base))), dtype=np.float32)
        x_query = np.asarray(_read_fvecs(Path(str(fvec_query))), dtype=np.float32)
        gt_raw = np.asarray(_read_ivecs(Path(str(fvec_gt))), dtype=np.int64)
        base_permute_seed = run_args.get("fvec_base_permute_seed")
        gt_label_offset = int(run_args.get("gt_label_offset", 0) or 0)
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

    raise ValueError("Unsupported run_args dataset source")


def _base5_features(raw: Dict[str, Any], eps: float = 1e-12) -> np.ndarray:
    """Compact 5D handcrafted feature vector used in LTT baselines."""

    d1 = float(raw["d1"])
    dk = float(raw["dk"])
    dk1 = float(raw["dk1"])
    topk = np.asarray(raw["dists_k"], dtype=np.float64)
    ef_explored = raw.get("ef_explored", None)
    if ef_explored is None:
        ef_explored = -1.0
    return np.asarray(
        [
            (dk1 - dk) / max(dk, eps),
            (dk1 - d1) / max(d1, eps),
            float(ef_explored),
            float(np.std(topk)),
            dk / max(d1, eps),
        ],
        dtype=np.float64,
    )


def _feature_vector(raw: Dict[str, Any], feature_mode: str) -> np.ndarray:
    """Feature dispatch for margin/base5/rich modes."""

    if feature_mode == "margin":
        return np.asarray([crc_core.margin_score(float(raw["dk"]), float(raw["dk1"]))], dtype=np.float64)
    if feature_mode == "base5":
        return _base5_features(raw)
    if feature_mode == "rich":
        return crc_core.build_feature_vector(raw)
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _fit_model(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_mode: str,
    model_mode: str,
) -> Dict[str, Any]:
    """Fit the configured score model on calibration-train partition."""

    if model_mode == "none":
        return {"kind": "identity"}

    uniq = np.unique(y_tr)
    if uniq.size < 2:
        return {"kind": "constant", "value": float(uniq[0]) if uniq.size else 0.0}

    if model_mode == "logreg":
        mu = np.mean(Z_tr, axis=0)
        sigma = np.std(Z_tr, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Zn = (Z_tr - mu) / sigma
        lr = LogisticRegression(max_iter=3000, class_weight="balanced")
        lr.fit(Zn, y_tr.astype(np.int64))
        return {
            "kind": "logreg",
            "mu": mu,
            "sigma": sigma,
            "coef": np.asarray(lr.coef_[0], dtype=np.float64),
            "intercept": float(lr.intercept_[0]),
        }

    # Keep "hgbdt" as a backward-compatible token, but use XGBoost implementation.
    if model_mode in ("hgbdt", "xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as e:
            raise ImportError(
                "xgboost is required for boosted-tree mode. Install it with: pip install xgboost"
            ) from e

        gb = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=0,
            n_jobs=1,
            verbosity=0,
        )
        gb.fit(Z_tr, y_tr.astype(np.int64))
        return {"kind": "xgb", "model": gb}

    raise ValueError(f"Unknown model_mode: {model_mode}")


def _predict_scores(model: Dict[str, Any], Z: np.ndarray) -> np.ndarray:
    """Predict pass scores from a fitted model payload."""

    kind = model["kind"]
    if kind == "identity":
        return np.asarray(Z[:, 0], dtype=np.float64)
    if kind == "constant":
        return np.full((Z.shape[0],), float(model["value"]), dtype=np.float64)
    if kind == "logreg":
        mu = np.asarray(model["mu"], dtype=np.float64)
        sigma = np.asarray(model["sigma"], dtype=np.float64)
        coef = np.asarray(model["coef"], dtype=np.float64)
        intercept = float(model["intercept"])
        Zn = (Z - mu) / sigma
        logits = Zn @ coef + intercept
        # stable sigmoid
        pos = logits >= 0
        out = np.empty_like(logits)
        out[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
        expz = np.exp(logits[~pos])
        out[~pos] = expz / (1.0 + expz)
        return out
    if kind == "xgb":
        gb = model["model"]
        return np.asarray(gb.predict_proba(Z)[:, 1], dtype=np.float64)
    raise ValueError(f"Unknown model kind: {kind}")


def _ltt_threshold(
    scores_tr: np.ndarray,
    scores_te: np.ndarray,
    recalls_te: np.ndarray,
    tau: float,
    alpha: float,
    epsilon: float,
    n_theta: int,
    use_bonferroni: bool,
    theta_select: str,
    theta_grid: str,
    theta_min: float | None,
    theta_max: float | None,
) -> Dict[str, Any]:
    """Run LTT over candidate theta values and choose theta_hat."""

    q = np.linspace(0.0, 1.0, n_theta)
    if theta_grid == "quantile":
        thetas = np.unique(np.quantile(scores_tr, q))
    elif theta_grid == "uniform":
        # Build the uniform grid directly on the requested interval when bounds
        # are provided, so |Theta| tracks n_theta instead of shrinking after clip.
        if theta_min is not None and theta_max is not None:
            lo = float(theta_min)
            hi = float(theta_max)
        elif theta_min is not None:
            lo = float(theta_min)
            hi = float(np.max(scores_tr))
        elif theta_max is not None:
            lo = float(np.min(scores_tr))
            hi = float(theta_max)
        else:
            lo = float(np.min(scores_tr))
            hi = float(np.max(scores_tr))
        thetas = np.unique(np.linspace(lo, hi, n_theta))
    elif theta_grid == "hybrid":
        lo = float(np.min(scores_tr))
        hi = float(np.max(scores_tr))
        tq = np.quantile(scores_tr, q)
        tu = np.linspace(lo, hi, n_theta)
        thetas = np.unique(np.concatenate([tq, tu], axis=0))
    else:
        raise ValueError(f"Unknown theta_grid: {theta_grid}")

    # Optional hard clipping of candidate-theta interval.
    # For uniform grid with explicit [theta_min, theta_max], we already build
    # directly in-range above, so skip post-clip to keep theta cardinality stable.
    if theta_grid != "uniform":
        if theta_min is not None:
            thetas = thetas[thetas >= float(theta_min)]
        if theta_max is not None:
            thetas = thetas[thetas <= float(theta_max)]

    if thetas.size == 0:
        return {
            "theta_hat": float("inf"),
            "reject_cutoff": 1.0,
            "theta_hat_p_value": float("nan"),
            "theta_tested_values": [],
            "h0_reject_theta_values": [],
            "h0_reject_p_values": [],
        }
    # If Bonferroni is disabled, use raw alpha as the cutoff.
    reject_cutoff = float(alpha) / float(thetas.size) if use_bonferroni else float(alpha)
    # Select threshold direction:
    # - max: strictest accepted threshold among those passing the test.
    # - min: loosest accepted threshold among those passing the test.
    if theta_select == "max":
        theta_iter = thetas[::-1]
    else:
        if theta_select == "min":
            theta_iter = thetas
        else:
            raise ValueError(f"Unknown theta_select: {theta_select}")

    pass_pairs: List[Tuple[float, float]] = []
    for theta in theta_iter:
        accepted = scores_te >= float(theta)
        m = int(np.sum(accepted))
        if m <= 0:
            continue
        x = int(np.sum(recalls_te[accepted] < float(tau)))
        pval = float(binom.cdf(x, m, float(epsilon)))
        if pval <= reject_cutoff:
            pass_pairs.append((float(theta), pval))
    if pass_pairs:
        if theta_select == "max":
            chosen_theta, chosen_p = max(pass_pairs, key=lambda t: t[0])
        else:
            chosen_theta, chosen_p = min(pass_pairs, key=lambda t: t[0])
    else:
        chosen_theta, chosen_p = float("inf"), float("nan")

    pass_pairs_sorted = sorted(pass_pairs, key=lambda t: t[0])
    return {
        "theta_hat": float(chosen_theta),
        "reject_cutoff": float(reject_cutoff),
        "theta_hat_p_value": float(chosen_p),
        "theta_tested_values": [float(x) for x in thetas.tolist()],
        "h0_reject_theta_values": [float(t) for t, _ in pass_pairs_sorted],
        "h0_reject_p_values": [float(p) for _, p in pass_pairs_sorted],
    }


def parse_args() -> argparse.Namespace:
    """Parse LTT alpha-sweep CLI options."""

    p = argparse.ArgumentParser(description="LTT alpha sweep with feature explorations on fixed test split.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run dir containing run_args.json for split/config")
    p.add_argument("--index-path", type=Path, default=None, help="Override HNSW index path (default: run_args.index_path)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--qcnt", type=int, default=100, help="Number of held-out test queries to evaluate")
    p.add_argument("--k", type=int, default=None, help="Override top-k (default: use run_args.k).")
    p.add_argument("--efs", type=int, default=None, help="Override ef_search (default: use run_args.efs).")
    p.add_argument(
        "--cal-size",
        type=int,
        default=None,
        help="Override calibration size (default: use run_args.cal_size).",
    )
    p.add_argument("--tau", type=float, default=0.90)
    p.add_argument("--epsilon", type=float, default=0.05, help="LTT failure-rate target among accepted")
    p.add_argument("--alphas", type=str, default="0.005,0.01,0.02,0.03,0.04,0.05,0.07,0.1")
    p.add_argument("--n-theta", type=int, default=100)
    p.add_argument("--ltt-train-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature-models", type=str, default="rich:logreg,base5:logreg,margin:none")
    p.add_argument("--no-bonferroni", action="store_true", help="Heuristic mode: use raw alpha instead of alpha/|Theta|")
    p.add_argument(
        "--theta-grid",
        type=str,
        default="quantile",
        choices=["quantile", "uniform", "hybrid"],
        help="How to construct Theta candidates for LTT testing.",
    )
    p.add_argument("--theta-min", type=float, default=None, help="Optional lower bound for candidate thetas.")
    p.add_argument("--theta-max", type=float, default=None, help="Optional upper bound for candidate thetas.")
    p.add_argument(
        "--theta-select",
        type=str,
        default="max",
        choices=["max", "min"],
        help="How to select among passing thresholds: max (strict) or min (permissive).",
    )
    p.add_argument("--rectify-c", type=float, default=3.91)
    p.add_argument("--rectify-expand", type=str, default="q", choices=["q", "x1"])
    p.add_argument("--rectify-use-pca", action="store_true")
    p.add_argument("--rectify-pca-prune-metric", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--rectify-stop-G", type=float, default=-1.0)
    p.add_argument("--rectify-drop-computed", action="store_true", default=True)
    p.add_argument("--rectify-prune-rem-budget-ann", action="store_true", default=False)
    p.add_argument(
        "--rectify-method",
        type=str,
        default="mbv",
        choices=["l0_dijkstra", "mbv", "sparse_l1"],
        help="Rectification backend: original L0 Dijkstra or sparse L1 pipeline.",
    )
    p.add_argument("--mbv-seed-count", type=int, default=0)
    p.add_argument("--mbv-seed-start", type=int, default=0)
    p.add_argument("--sparse-l1-k", type=int, default=100, help="Seed count from L1 ANN before sparse L1 Dijkstra.")
    p.add_argument("--sparse-l1-ef", type=int, default=200, help="ef used for sparse L1 ANN and precompute mapping.")
    p.add_argument("--sparse-l1-t", type=float, default=3.5, help="Stretch factor for sparse L1 Dijkstra.")
    p.add_argument("--sparse-precompute-k-trace", type=int, default=1, help="k_trace when building L0->L1 lookup.")
    p.add_argument(
        "--sparse-map-dir",
        type=Path,
        default=Path("artifacts/sparse_l1_maps"),
        help="Directory for persisted sparse L0->L1 map caches.",
    )
    p.add_argument("--sparse-map-path", type=Path, default=None, help="Optional explicit sparse map (.npz) path.")
    p.add_argument("--sparse-pca-cache", type=Path, default=None, help="Optional explicit PCA cache (.npz) path.")
    p.add_argument("--sparse-pca-dims", type=int, default=-1, help="PCA dimensions override for sparse mode.")
    p.add_argument(
        "--sparse-subtract-ann-time",
        dest="sparse_subtract_ann_time",
        action="store_true",
        help="Subtract sparse stage-1 ANN from sparse total timing.",
    )
    p.add_argument(
        "--no-sparse-subtract-ann-time",
        dest="sparse_subtract_ann_time",
        action="store_false",
        help="Keep sparse stage-1 ANN inside sparse total timing.",
    )
    p.set_defaults(sparse_subtract_ann_time=True)
    return p.parse_args()


def main() -> None:
    """Run LTT alpha sweep and write summary/per-query CSVs."""

    args = parse_args()
    run_args = json.loads((args.run_dir / "run_args.json").read_text())
    tau = float(args.tau)
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    specs: List[Tuple[str, str]] = []
    for tok in args.feature_models.split(","):
        tok = tok.strip()
        if not tok:
            continue
        feature_mode, model_mode = tok.split(":")
        specs.append((feature_mode.strip(), model_mode.strip()))

    x_train, x_query, gt = _load_dataset_from_run_args(run_args)
    hnswlib = sparse.import_hnswlib_with_optional_override()
    k = int(run_args["k"]) if args.k is None else int(args.k)
    efs = int(run_args["efs"]) if args.efs is None else int(args.efs)
    index_path = str(run_args["index_path"]) if args.index_path is None else str(args.index_path)

    rng = np.random.default_rng(int(run_args["seed"]))
    perm = rng.permutation(x_query.shape[0])
    if run_args["qcnt"] is not None:
        perm = perm[: min(int(run_args["qcnt"]), perm.size)]
    target_cal = int(run_args["cal_size"]) if args.cal_size is None else int(args.cal_size)
    cal_n = min(target_cal, perm.size - 1)
    cal_idx = perm[:cal_n]
    test_idx = perm[cal_n:]
    test_idx = test_idx[: min(int(args.qcnt), test_idx.size)]

    X_cal = np.asarray(x_query[cal_idx], dtype=np.float32)
    GT_cal = np.asarray(gt[cal_idx, :k], dtype=np.int64)
    X_test = np.asarray(x_query[test_idx], dtype=np.float32)
    GT_test = np.asarray(gt[test_idx, :k], dtype=np.int64)

    idx = hnswlib.Index(space=run_args["space"], dim=int(x_train.shape[1]))
    idx.load_index(index_path, max_elements=int(x_train.shape[0]))
    idx.set_ef(int(efs))
    if hasattr(idx, "ensure_level0_edge_weight_cache"):
        idx.ensure_level0_edge_weight_cache()

    if args.rectify_method == "sparse_l1":
        dataset_tag = sparse.infer_dataset_tag(index_path=index_path, run_args=run_args)
        if bool(args.rectify_use_pca):
            pca_dims = int(args.sparse_pca_dims)
            if pca_dims <= 0:
                pca_dims = sparse.default_pca_dims_for_dataset(dataset_tag)
            pca_cache = (
                args.sparse_pca_cache
                if args.sparse_pca_cache is not None
                else sparse.default_pca_cache_path(dataset_tag=dataset_tag, pca_dims=pca_dims)
            )
            sparse.load_pca_into_index(idx, Path(pca_cache))
        map_path = (
            args.sparse_map_path
            if args.sparse_map_path is not None
            else sparse.default_sparse_map_path(
                dataset_tag=dataset_tag,
                index_path=index_path,
                l1_ef=int(args.sparse_l1_ef),
                map_dir=Path(args.sparse_map_dir),
            )
        )
        sparse.ensure_sparse_lookup(
            idx,
            map_path=Path(map_path),
            precompute_k_trace=int(args.sparse_precompute_k_trace),
        )

    # 1) Precompute raw ANN + recalls on calibration.
    cal_raw: List[Dict[str, Any]] = []
    cal_recall = np.zeros(len(X_cal), dtype=np.float64)
    for i in range(len(X_cal)):
        raw = crc_core.default_query_index(idx, X_cal[i], k, int(efs))
        cal_raw.append(raw)
        cal_recall[i] = float(crc_core.compute_recall(np.asarray(raw["labels_k"], dtype=np.int64), GT_cal[i][:k]))

    # 2) Split calibration into LTT train/test.
    rng_ltt = np.random.default_rng(int(args.seed))
    cperm = rng_ltt.permutation(len(X_cal))
    tr_n = max(1, min(len(X_cal) - 1, int(round(len(X_cal) * float(args.ltt_train_frac)))))
    tr_idx = cperm[:tr_n]
    te_idx = cperm[tr_n:]

    # 3) Precompute test ANN + rectify once.
    n_test = len(X_test)
    test_raw: List[Dict[str, Any]] = []
    hnsw_recall = np.zeros(n_test, dtype=np.float64)
    hnsw_time_ns = np.zeros(n_test, dtype=np.int64)
    rectify_recall = np.zeros(n_test, dtype=np.float64)
    rectify_time_ns = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        q = X_test[i]
        gt_k = GT_test[i][:k]

        t0 = time.perf_counter_ns()
        raw = crc_core.default_query_index(idx, q, k, int(efs))
        t1 = time.perf_counter_ns()
        hnsw_time_ns[i] = int(t1 - t0)
        test_raw.append(raw)

        labels_k = np.asarray(raw["labels_k"], dtype=np.int64)
        dists_k = np.asarray(raw["dists_k"], dtype=np.float64)
        hnsw_recall[i] = float(crc_core.compute_recall(labels_k, gt_k))

        if args.rectify_method == "l0_dijkstra" and hasattr(idx, "rectify_from_stage1_ex"):
            out_labels, _out_dists, stats = idx.rectify_from_stage1_ex(
                q,
                labels_k,
                dists_k,
                c=float(args.rectify_c),
                expand=args.rectify_expand,
                use_pca=bool(args.rectify_use_pca),
                pca_prune_metric=args.rectify_pca_prune_metric,
                stop_G=float(args.rectify_stop_G),
                euclidean=True,
                drop_computed=bool(args.rectify_drop_computed),
                gt_labels=gt_k,
            )
            rectify_ns = int(stats["total_time_ns"])
        elif args.rectify_method == "mbv" or not hasattr(idx, "sparse_l1_full_query_ex"):
            out_labels, _out_dists, stats, rectify_ns = sparse.run_mbv_rectify_query(
                index=idx,
                q=q,
                gt_k=gt_k,
                cfg=sparse.MBVRectifyConfig(
                    k=int(k),
                    t=float(args.rectify_c),
                    seed_count=int(args.mbv_seed_count),
                    seed_start=int(args.mbv_seed_start),
                    subtract_ann_time=True,
                ),
            )
        else:
            out_labels, _out_dists, stats, rectify_ns = sparse.run_sparse_rectify_query(
                index=idx,
                q=q,
                gt_k=gt_k,
                cfg=sparse.SparseRectifyConfig(
                    k=int(k),
                    l1_k=int(args.sparse_l1_k),
                    l1_t=float(args.sparse_l1_t),
                    l1_ef=int(args.sparse_l1_ef),
                    use_pca=bool(args.rectify_use_pca),
                    subtract_ann_time=bool(args.sparse_subtract_ann_time),
                ),
            )
        rectify_recall[i] = float(crc_core.compute_recall(np.asarray(out_labels, dtype=np.int64), gt_k))
        rectify_time_ns[i] = int(rectify_ns)

    summary_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []

    for feature_mode, model_mode in specs:
        Z_cal = np.vstack([_feature_vector(r, feature_mode) for r in cal_raw])
        y_cal = (cal_recall >= tau).astype(np.int64)
        model = _fit_model(Z_cal[tr_idx], y_cal[tr_idx], feature_mode, model_mode)
        scores_tr = _predict_scores(model, Z_cal[tr_idx])
        scores_te = _predict_scores(model, Z_cal[te_idx])
        recalls_te = cal_recall[te_idx]

        Z_test = np.vstack([_feature_vector(r, feature_mode) for r in test_raw])

        # Measure pure scoring time on test queries for this feature/model.
        t0 = time.perf_counter_ns()
        test_scores = _predict_scores(model, Z_test)
        t1 = time.perf_counter_ns()
        score_time_per_query_ns = float((t1 - t0) / max(1, n_test))

        for alpha in alphas:
            ltt = _ltt_threshold(
                scores_tr=scores_tr,
                scores_te=scores_te,
                recalls_te=recalls_te,
                tau=tau,
                alpha=float(alpha),
                epsilon=float(args.epsilon),
                n_theta=int(args.n_theta),
                use_bonferroni=(not bool(args.no_bonferroni)),
                theta_select=str(args.theta_select),
                theta_grid=str(args.theta_grid),
                theta_min=(None if args.theta_min is None else float(args.theta_min)),
                theta_max=(None if args.theta_max is None else float(args.theta_max)),
            )
            theta_hat = float(ltt["theta_hat"])
            reject_cutoff = float(ltt["reject_cutoff"])
            theta_hat_p_value = float(ltt["theta_hat_p_value"])
            theta_tested_values = json.dumps(ltt["theta_tested_values"], separators=(",", ":"))
            h0_reject_theta_values = json.dumps(ltt["h0_reject_theta_values"], separators=(",", ":"))
            h0_reject_p_values = json.dumps(ltt["h0_reject_p_values"], separators=(",", ":"))
            h0_reject_count = int(len(ltt["h0_reject_theta_values"]))

            accept = test_scores >= theta_hat
            pred_not_rectify = accept
            actual_not_rectify = hnsw_recall >= tau
            tp = int(np.sum(pred_not_rectify & actual_not_rectify))
            fp = int(np.sum(pred_not_rectify & (~actual_not_rectify)))
            tn = int(np.sum((~pred_not_rectify) & (~actual_not_rectify)))
            fn = int(np.sum((~pred_not_rectify) & actual_not_rectify))

            final_recall = np.where(accept, hnsw_recall, rectify_recall)
            combined_time = hnsw_time_ns.astype(np.float64) + score_time_per_query_ns + np.where(~accept, rectify_time_ns, 0.0)

            rectify_calls = int(np.sum(~accept))
            summary_rows.append(
                {
                    "feature_mode": feature_mode,
                    "model_mode": model_mode,
                    "tau": tau,
                    "rectify_method": str(args.rectify_method),
                    "mbv_seed_count": int(args.mbv_seed_count),
                    "mbv_seed_start": int(args.mbv_seed_start),
                    "epsilon": float(args.epsilon),
                    "alpha": float(alpha),
                    "n_theta": int(args.n_theta),
                    "theta_grid": str(args.theta_grid),
                    "theta_min": (None if args.theta_min is None else float(args.theta_min)),
                    "theta_max": (None if args.theta_max is None else float(args.theta_max)),
                    "ltt_train_frac": float(args.ltt_train_frac),
                    "theta_hat": float(theta_hat),
                    "theta_hat_p_value": float(theta_hat_p_value),
                    "reject_cutoff": float(reject_cutoff),
                    "theta_tested_values": theta_tested_values,
                    "theta_tested_count": int(len(ltt["theta_tested_values"])),
                    "h0_reject_theta_values": h0_reject_theta_values,
                    "h0_reject_p_values": h0_reject_p_values,
                    "h0_reject_count": h0_reject_count,
                    "n_test_eval": int(n_test),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "rectify_calls": rectify_calls,
                    "accept_rate": float(np.mean(accept)),
                    "hnsw_mean_recall": float(np.mean(hnsw_recall)),
                    "final_mean_recall": float(np.mean(final_recall)),
                    "hnsw_mean_time_ns": float(np.mean(hnsw_time_ns)),
                    "score_mean_time_ns": float(score_time_per_query_ns),
                    "rectify_mean_time_ns_over_all": float(np.mean(np.where(~accept, rectify_time_ns, 0.0))),
                    "combined_mean_time_ns": float(np.mean(combined_time)),
                    "combined_over_hnsw_ratio": float(np.mean(combined_time) / max(1e-30, np.mean(hnsw_time_ns))),
                }
            )

            for qi in range(n_test):
                per_query_rows.append(
                    {
                        "feature_mode": feature_mode,
                        "model_mode": model_mode,
                        "alpha": float(alpha),
                        "rectify_method": str(args.rectify_method),
                        "mbv_seed_count": int(args.mbv_seed_count),
                        "mbv_seed_start": int(args.mbv_seed_start),
                        "qid": int(qi),
                        "score": float(test_scores[qi]),
                        "theta_hat": float(theta_hat),
                        "theta_hat_p_value": float(theta_hat_p_value),
                        "theta_tested_values": theta_tested_values,
                        "h0_reject_theta_values": h0_reject_theta_values,
                        "h0_reject_p_values": h0_reject_p_values,
                        "h0_reject_count": h0_reject_count,
                        "pred_not_rectify": int(accept[qi]),
                        "actual_not_rectify": int(actual_not_rectify[qi]),
                        "hnsw_recall": float(hnsw_recall[qi]),
                        "final_recall": float(final_recall[qi]),
                        "hnsw_time_ns": int(hnsw_time_ns[qi]),
                        "score_time_ns": float(score_time_per_query_ns),
                        "rectify_time_ns": int(rectify_time_ns[qi] if not accept[qi] else 0),
                        "combined_time_ns": float(combined_time[qi]),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "ltt_alpha_feature_sweep_summary.csv"
    detail_csv = args.out_dir / "ltt_alpha_feature_sweep_per_query.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with detail_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "per_query_csv": str(detail_csv),
                "num_summary_rows": len(summary_rows),
                "num_per_query_rows": len(per_query_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
