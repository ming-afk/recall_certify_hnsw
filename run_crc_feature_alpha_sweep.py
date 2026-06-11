#!/usr/bin/env python3
from __future__ import annotations

"""CRC alpha sweep on a fixed split/index with optional augmented scorer.

Inputs are anchored by a previous `run_crc_experiment.py` output directory so
all split/index defaults remain reproducible across reruns.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import crc_core
import run_ltt_feature_alpha_sweep as ltt
import sparse_rectify as sparse


def _predict_crc_scores(use_augmented: bool, model_kind: str, model_params: Dict[str, Any], raws: List[Dict[str, Any]]) -> np.ndarray:
    """Score queries using either margin-only CRC or augmented model."""

    if not use_augmented:
        return np.asarray([crc_core.margin_score(float(r["dk"]), float(r["dk1"])) for r in raws], dtype=np.float64)
    z = np.vstack([crc_core.build_feature_vector(r) for r in raws])
    return np.asarray([crc_core.augmented_score(model_kind, model_params, z[i]) for i in range(z.shape[0])], dtype=np.float64)


def _precompute_theta_risks(scores: np.ndarray, losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute risk(theta) for all unique score values."""

    # For each unique theta (ascending), compute empirical CRC risk:
    # risk(theta) = sum_{score>=theta} loss / (count_{score>=theta} + 1)
    order = np.argsort(scores)
    s = np.asarray(scores[order], dtype=np.float64)
    l = np.asarray(losses[order], dtype=np.float64)
    n = s.size
    if n == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    thetas, first_idx = np.unique(s, return_index=True)
    suffix_loss = np.cumsum(l[::-1])[::-1]
    suffix_count = (n - first_idx).astype(np.float64)
    risks = suffix_loss[first_idx] / (suffix_count + 1.0)
    return np.asarray(thetas, dtype=np.float64), np.asarray(risks, dtype=np.float64)


def _theta_for_alpha(alpha: float, tau: float, n_cal: int, thetas: np.ndarray, risks: np.ndarray) -> Tuple[float, float]:
    """Map alpha to the first threshold satisfying CRC bound."""

    target_bound = float(alpha) * (1.0 - float(tau)) * (float(n_cal) / (float(n_cal) + 1.0))
    idx = np.where(risks <= target_bound)[0]
    if idx.size == 0:
        return float("inf"), float(target_bound)
    return float(thetas[int(idx[0])]), float(target_bound)


def parse_args() -> argparse.Namespace:
    """Parse alpha-sweep CLI options."""

    p = argparse.ArgumentParser(description="CRC alpha sweep with fixed split and rectify evaluation.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run dir containing run_args.json")
    p.add_argument("--index-path", type=Path, default=None, help="Override index path (default: run_args.index_path)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--qcnt", type=int, default=100)
    p.add_argument("--cal-size", type=int, default=None, help="Override calibration size (default: run_args.cal_size)")
    p.add_argument("--k", type=int, default=None, help="Override k (default: run_args.k)")
    p.add_argument("--efs", type=int, default=None, help="Override ef_search (default: run_args.efs)")
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--alphas", type=str, default="0.5,1.0e-1,1.0e-2,1.0e-3")
    p.add_argument("--alpha-mode", type=str, default="explicit", choices=["explicit", "adaptive"])
    p.add_argument("--alpha-min", type=float, default=1.0e-180)
    p.add_argument("--alpha-max", type=float, default=0.5)
    p.add_argument("--alpha-cap", type=int, default=240, help="Maximum adaptive alpha count before jitter expansion.")
    p.add_argument(
        "--alpha-jitter",
        type=str,
        default="0.98,1.0,1.02",
        help="Comma-separated multiplicative jitter factors around adaptive transition alphas.",
    )
    p.add_argument("--use-augmented", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ensure-legacy-edge-cache",
        action="store_true",
        default=True,
        help="For legacy indices without inline edge weights, build the in-memory L0 edge cache before timing rectification.",
    )
    p.add_argument(
        "--no-ensure-legacy-edge-cache",
        dest="ensure_legacy_edge_cache",
        action="store_false",
        help="Disable legacy L0 edge-weight cache warmup.",
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
        help="Rectification backend: original L0 Dijkstra, MBV from stage-1 ANN, or sparse L1 pipeline.",
    )
    p.add_argument("--mbv-seed-count", type=int, default=0, help="Top-k ANN seeds used to initialize MBV; 0 means all.")
    p.add_argument("--mbv-seed-start", type=int, default=0, help="Start offset inside the ANN top-k seed list for MBV.")
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


def _adaptive_alphas_from_risks(
    risks: np.ndarray,
    tau: float,
    n_cal: int,
    alpha_min: float,
    alpha_max: float,
    alpha_cap: int,
    jitter_factors: List[float],
) -> List[float]:
    """Generate alpha candidates from risk transitions plus local jitter."""

    coef = (1.0 - float(tau)) * (float(n_cal) / (float(n_cal) + 1.0))
    if coef <= 0:
        return [float(alpha_max), float(alpha_min)]

    base = np.asarray(risks, dtype=np.float64) / float(coef)
    base = base[np.isfinite(base)]
    base = base[(base > 0.0)]
    base = np.clip(base, float(alpha_min), float(alpha_max))
    if base.size == 0:
        return [float(alpha_max), float(alpha_min)]
    base = np.unique(base)

    if base.size > int(alpha_cap):
        logs = np.log10(base)
        targets = np.linspace(float(logs.min()), float(logs.max()), int(alpha_cap))
        idx = np.searchsorted(logs, targets)
        idx = np.clip(idx, 0, base.size - 1)
        base = np.unique(base[idx])

    cand = [base]
    for jf in jitter_factors:
        cand.append(np.clip(base * float(jf), float(alpha_min), float(alpha_max)))
    arr = np.unique(np.concatenate(cand + [np.asarray([float(alpha_min), float(alpha_max)], dtype=np.float64)]))
    arr = arr[np.isfinite(arr)]
    arr = np.asarray(sorted(arr.tolist(), reverse=True), dtype=np.float64)
    return [float(x) for x in arr.tolist()]


def main() -> None:
    """Run fixed-split CRC alpha sweep and write summary/per-query CSVs."""

    args = parse_args()
    run_args = json.loads((args.run_dir / "run_args.json").read_text())

    k = int(run_args["k"]) if args.k is None else int(args.k)
    efs = int(run_args["efs"]) if args.efs is None else int(args.efs)
    cal_size = int(run_args["cal_size"]) if args.cal_size is None else int(args.cal_size)
    tau = float(args.tau)
    index_path = str(run_args["index_path"]) if args.index_path is None else str(args.index_path)
    if args.alpha_mode == "explicit":
        alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        alphas = []

    x_train, x_query, gt = ltt._load_dataset_from_run_args(run_args)
    hnswlib = sparse.import_hnswlib_with_optional_override()

    rng = np.random.default_rng(int(run_args.get("seed", args.seed)))
    perm = rng.permutation(x_query.shape[0])
    if run_args["qcnt"] is not None:
        perm = perm[: min(int(run_args["qcnt"]), perm.size)]
    cal_n = min(int(cal_size), perm.size - 1)
    cal_idx = perm[:cal_n]
    test_idx = perm[cal_n:]
    test_idx = test_idx[: min(int(args.qcnt), test_idx.size)]

    X_cal = np.asarray(x_query[cal_idx], dtype=np.float32)
    GT_cal = np.asarray(gt[cal_idx, :k], dtype=np.int64)
    X_test = np.asarray(x_query[test_idx], dtype=np.float32)
    GT_test = np.asarray(gt[test_idx, :k], dtype=np.int64)

    idx = hnswlib.Index(space=str(run_args["space"]), dim=int(x_train.shape[1]))
    idx.load_index(index_path, max_elements=int(x_train.shape[0]))
    idx.set_ef(int(efs))
    legacy_edge_cache_stats = None
    if bool(args.ensure_legacy_edge_cache) and hasattr(idx, "ensure_level0_edge_weight_cache"):
        legacy_edge_cache_stats = dict(idx.ensure_level0_edge_weight_cache())

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

    # 1) Calibration precompute.
    cal_raw: List[Dict[str, Any]] = []
    cal_recall = np.zeros(len(X_cal), dtype=np.float64)
    for i in range(len(X_cal)):
        raw = crc_core.default_query_index(idx, X_cal[i], k, int(efs))
        cal_raw.append(raw)
        cal_recall[i] = float(crc_core.compute_recall(np.asarray(raw["labels_k"], dtype=np.int64), GT_cal[i][:k]))
    cal_losses = np.asarray([crc_core.compute_crc_loss(float(r), tau) for r in cal_recall], dtype=np.float64)

    if args.use_augmented:
        Z_cal = np.vstack([crc_core.build_feature_vector(r) for r in cal_raw])
        model_kind, model_params = crc_core.fit_augmented_score_model(Z_cal, cal_recall, tau)
    else:
        model_kind, model_params = "margin", {}
    cal_scores = _predict_crc_scores(bool(args.use_augmented), model_kind, model_params, cal_raw)
    theta_vals, risk_vals = _precompute_theta_risks(cal_scores, cal_losses)

    if args.alpha_mode == "adaptive":
        jitter = [float(x.strip()) for x in str(args.alpha_jitter).split(",") if x.strip()]
        alphas = _adaptive_alphas_from_risks(
            risks=risk_vals,
            tau=tau,
            n_cal=len(cal_scores),
            alpha_min=float(args.alpha_min),
            alpha_max=float(args.alpha_max),
            alpha_cap=int(args.alpha_cap),
            jitter_factors=jitter,
        )

    # 2) Test precompute (ANN + rectify once).
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

    # 3) Measure certifier scoring overhead once.
    t0 = time.perf_counter_ns()
    test_scores = _predict_crc_scores(bool(args.use_augmented), model_kind, model_params, test_raw)
    t1 = time.perf_counter_ns()
    score_time_per_query_ns = float((t1 - t0) / max(1, n_test))

    actual_not_rectify = hnsw_recall >= tau

    summary_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        theta_hat, target_bound = _theta_for_alpha(alpha=float(alpha), tau=tau, n_cal=len(cal_scores), thetas=theta_vals, risks=risk_vals)
        accept = test_scores >= theta_hat
        pred_not_rectify = accept

        tp = int(np.sum(pred_not_rectify & actual_not_rectify))
        fp = int(np.sum(pred_not_rectify & (~actual_not_rectify)))
        tn = int(np.sum((~pred_not_rectify) & (~actual_not_rectify)))
        fn = int(np.sum((~pred_not_rectify) & actual_not_rectify))

        final_recall = np.where(accept, hnsw_recall, rectify_recall)
        combined_time = hnsw_time_ns.astype(np.float64) + score_time_per_query_ns + np.where(~accept, rectify_time_ns, 0.0)

        summary_rows.append(
            {
                "model_kind": model_kind,
                "alpha_mode": str(args.alpha_mode),
                "tau": tau,
                "rectify_method": str(args.rectify_method),
                "mbv_seed_count": int(args.mbv_seed_count),
                "mbv_seed_start": int(args.mbv_seed_start),
                "alpha": float(alpha),
                "target_bound": float(target_bound),
                "theta_hat": float(theta_hat),
                "n_test_eval": int(n_test),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "rectify_calls": int(np.sum(~accept)),
                "accept_rate": float(np.mean(accept)),
                "hnsw_mean_recall": float(np.mean(hnsw_recall)),
                "final_mean_recall": float(np.mean(final_recall)),
                "hnsw_mean_time_ns": float(np.mean(hnsw_time_ns)),
                "crc_mean_time_ns": float(score_time_per_query_ns),
                "rectify_mean_time_ns_over_all": float(np.mean(np.where(~accept, rectify_time_ns, 0.0))),
                "combined_mean_time_ns": float(np.mean(combined_time)),
                "combined_over_hnsw_ratio": float(np.mean(combined_time) / max(1e-30, np.mean(hnsw_time_ns))),
            }
        )

        for qi in range(n_test):
            per_query_rows.append(
                {
                    "alpha": float(alpha),
                    "rectify_method": str(args.rectify_method),
                    "mbv_seed_count": int(args.mbv_seed_count),
                    "mbv_seed_start": int(args.mbv_seed_start),
                    "qid": int(qi),
                    "score": float(test_scores[qi]),
                    "theta_hat": float(theta_hat),
                    "pred_not_rectify": int(accept[qi]),
                    "actual_not_rectify": int(actual_not_rectify[qi]),
                    "hnsw_recall": float(hnsw_recall[qi]),
                    "final_recall": float(final_recall[qi]),
                    "hnsw_time_ns": int(hnsw_time_ns[qi]),
                    "crc_time_ns": float(score_time_per_query_ns),
                    "rectify_time_ns": int(rectify_time_ns[qi] if not accept[qi] else 0),
                    "combined_time_ns": float(combined_time[qi]),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "crc_alpha_sweep_summary.csv"
    per_query_csv = args.out_dir / "crc_alpha_sweep_per_query.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with per_query_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "per_query_csv": str(per_query_csv),
                "legacy_edge_cache": legacy_edge_cache_stats,
                "num_summary_rows": len(summary_rows),
                "num_per_query_rows": len(per_query_rows),
                "num_alpha": len(alphas),
                "alpha_min": float(min(alphas) if alphas else float("nan")),
                "alpha_max": float(max(alphas) if alphas else float("nan")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
