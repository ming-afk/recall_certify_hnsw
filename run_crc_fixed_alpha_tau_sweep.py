#!/usr/bin/env python3
from __future__ import annotations

"""CRC tau sweep with fixed alpha on a reproducible split/index.

This script reuses run metadata (`run_args.json`) from a prior CRC run and
evaluates how decisions/latency move as tau changes.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hnswlib  # type: ignore
import numpy as np

import crc_core
import run_ltt_feature_alpha_sweep as ltt
import sparse_rectify as sparse


def _predict_crc_scores(use_augmented: bool, model_kind: str, model_params: Dict[str, Any], raws: List[Dict[str, Any]]) -> np.ndarray:
    """Score queries using margin-only or augmented CRC model."""

    if not use_augmented:
        return np.asarray([crc_core.margin_score(float(r["dk"]), float(r["dk1"])) for r in raws], dtype=np.float64)
    z = np.vstack([crc_core.build_feature_vector(r) for r in raws])
    return np.asarray([crc_core.augmented_score(model_kind, model_params, z[i]) for i in range(z.shape[0])], dtype=np.float64)


def _precompute_theta_risks(scores: np.ndarray, losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute empirical CRC risk for each candidate threshold."""

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
    """Return threshold selected by fixed alpha at a given tau."""

    target_bound = float(alpha) * (1.0 - float(tau)) * (float(n_cal) / (float(n_cal) + 1.0))
    idx = np.where(risks <= target_bound)[0]
    if idx.size == 0:
        return float("inf"), float(target_bound)
    return float(thetas[int(idx[0])]), float(target_bound)


def _parse_taus(s: str) -> List[float]:
    """Parse comma-separated tau values."""

    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    """Parse fixed-alpha tau-sweep CLI options."""

    p = argparse.ArgumentParser(description="CRC fixed-alpha tau sweep on fixed split/index.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("crc_runs/sift_crc_currentfeat_tau90_a0p13_m16_efc100_efs100_k100_cal9000_q10000"),
    )
    p.add_argument("--index-path", type=Path, default=None, help="Optional explicit index path; defaults to run_args.index_path")
    p.add_argument("--out-dir", type=Path, default=Path("crc_runs/sift_crc_fixed_alpha_tau_sweep_M16_efs100_k100"))
    p.add_argument("--qcnt", type=int, default=100)
    p.add_argument("--cal-size", type=int, default=5000)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--efs", type=int, default=100)
    p.add_argument("--taus", type=str, default="0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99")
    p.add_argument("--fixed-alpha", type=float, default=None, help="If omitted, use run_args.alpha.")
    p.add_argument("--use-augmented", action="store_true", default=True)

    p.add_argument("--rectify-c", type=float, default=3.91)
    p.add_argument("--rectify-expand", type=str, default="q", choices=["q", "x1"])
    p.add_argument("--rectify-use-pca", action="store_true")
    p.add_argument("--rectify-pca-prune-metric", type=str, default="linf", choices=["linf", "l2"])
    p.add_argument("--rectify-stop-G", type=float, default=-1.0)
    p.add_argument("--rectify-drop-computed", action="store_true", default=True)
    p.add_argument("--rectify-prune-rem-budget-ann", action="store_true", default=False)
    p.add_argument("--rectify-method", type=str, default="mbv", choices=["l0_dijkstra", "mbv"])
    p.add_argument("--mbv-seed-count", type=int, default=0)
    p.add_argument("--mbv-seed-start", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Run tau sweep and write CSV + plot artifacts."""

    args = parse_args()
    run_args = json.loads((args.run_dir / "run_args.json").read_text())
    taus = _parse_taus(str(args.taus))
    fixed_alpha = float(args.fixed_alpha) if args.fixed_alpha is not None else float(run_args.get("alpha", 0.13))

    x_train, x_query, gt = ltt._load_dataset_from_run_args(run_args)
    k = int(args.k)
    efs = int(args.efs)
    index_path = str(args.index_path) if args.index_path is not None else str(run_args["index_path"])

    rng = np.random.default_rng(int(run_args.get("seed", 0)))
    perm = rng.permutation(x_query.shape[0])
    if run_args["qcnt"] is not None:
        perm = perm[: min(int(run_args["qcnt"]), perm.size)]
    cal_n = min(int(args.cal_size), perm.size - 1)
    cal_idx = perm[:cal_n]
    test_idx = perm[cal_n:]
    test_idx = test_idx[: min(int(args.qcnt), test_idx.size)]

    x_cal = np.asarray(x_query[cal_idx], dtype=np.float32)
    gt_cal = np.asarray(gt[cal_idx, :k], dtype=np.int64)
    x_test = np.asarray(x_query[test_idx], dtype=np.float32)
    gt_test = np.asarray(gt[test_idx, :k], dtype=np.int64)

    idx = hnswlib.Index(space=str(run_args["space"]), dim=int(x_train.shape[1]))
    idx.load_index(index_path, max_elements=int(x_train.shape[0]))
    idx.set_ef(int(efs))
    if hasattr(idx, "ensure_level0_edge_weight_cache"):
        idx.ensure_level0_edge_weight_cache()

    # 1) Calibration precompute.
    cal_raw: List[Dict[str, Any]] = []
    cal_recall = np.zeros(len(x_cal), dtype=np.float64)
    for i in range(len(x_cal)):
        raw = crc_core.default_query_index(idx, x_cal[i], k, int(efs))
        cal_raw.append(raw)
        cal_recall[i] = float(crc_core.compute_recall(np.asarray(raw["labels_k"], dtype=np.int64), gt_cal[i][:k]))

    if args.use_augmented:
        z_cal = np.vstack([crc_core.build_feature_vector(r) for r in cal_raw])
        model_kind, model_params = crc_core.fit_augmented_score_model(z_cal, cal_recall, float(np.median(taus)))
    else:
        model_kind, model_params = "margin", {}
    cal_scores = _predict_crc_scores(bool(args.use_augmented), model_kind, model_params, cal_raw)

    # 2) Test ANN + rectify precompute.
    n_test = len(x_test)
    test_raw: List[Dict[str, Any]] = []
    hnsw_recall = np.zeros(n_test, dtype=np.float64)
    hnsw_time_ns = np.zeros(n_test, dtype=np.int64)
    rectify_recall = np.zeros(n_test, dtype=np.float64)
    rectify_time_ns = np.zeros(n_test, dtype=np.int64)
    for i in range(n_test):
        q = x_test[i]
        gt_k = gt_test[i][:k]

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
                expand=str(args.rectify_expand),
                use_pca=bool(args.rectify_use_pca),
                pca_prune_metric=str(args.rectify_pca_prune_metric),
                stop_G=float(args.rectify_stop_G),
                euclidean=True,
                drop_computed=bool(args.rectify_drop_computed),
                gt_labels=gt_k,
            )
            rectify_ns = int(stats["total_time_ns"])
        else:
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
        rectify_recall[i] = float(crc_core.compute_recall(np.asarray(out_labels, dtype=np.int64), gt_k))
        rectify_time_ns[i] = int(rectify_ns)

    # 3) Measure certifier scoring overhead once.
    t0 = time.perf_counter_ns()
    test_scores = _predict_crc_scores(bool(args.use_augmented), model_kind, model_params, test_raw)
    t1 = time.perf_counter_ns()
    score_time_per_query_ns = float((t1 - t0) / max(1, n_test))

    summary_rows: List[Dict[str, Any]] = []
    per_query_rows: List[Dict[str, Any]] = []
    for tau in taus:
        cal_losses = np.asarray([crc_core.compute_crc_loss(float(r), float(tau)) for r in cal_recall], dtype=np.float64)
        theta_vals, risk_vals = _precompute_theta_risks(cal_scores, cal_losses)
        theta_hat, target_bound = _theta_for_alpha(
            alpha=float(fixed_alpha),
            tau=float(tau),
            n_cal=len(cal_scores),
            thetas=theta_vals,
            risks=risk_vals,
        )

        accept = test_scores >= theta_hat
        actual_not_rectify = hnsw_recall >= float(tau)
        pred_not_rectify = accept
        tp = int(np.sum(pred_not_rectify & actual_not_rectify))
        fp = int(np.sum(pred_not_rectify & (~actual_not_rectify)))
        tn = int(np.sum((~pred_not_rectify) & (~actual_not_rectify)))
        fn = int(np.sum((~pred_not_rectify) & actual_not_rectify))

        final_recall = np.where(accept, hnsw_recall, rectify_recall)
        combined_time = hnsw_time_ns.astype(np.float64) + score_time_per_query_ns + np.where(~accept, rectify_time_ns, 0.0)

        summary_rows.append(
            {
                "tau": float(tau),
                "rectify_method": str(args.rectify_method),
                "mbv_seed_count": int(args.mbv_seed_count),
                "mbv_seed_start": int(args.mbv_seed_start),
                "fixed_alpha": float(fixed_alpha),
                "theta_hat": float(theta_hat),
                "target_bound": float(target_bound),
                "n_test_eval": int(n_test),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accept_rate": float(np.mean(accept)),
                "rectify_calls": int(np.sum(~accept)),
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
                    "tau": float(tau),
                    "rectify_method": str(args.rectify_method),
                    "mbv_seed_count": int(args.mbv_seed_count),
                    "mbv_seed_start": int(args.mbv_seed_start),
                    "fixed_alpha": float(fixed_alpha),
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

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "crc_fixed_alpha_tau_sweep_summary.csv"
    per_query_csv = out / "crc_fixed_alpha_tau_sweep_per_query.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with per_query_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    summary = {
        "tau_values": taus,
        "fixed_alpha": float(fixed_alpha),
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
        "n_test_eval": int(n_test),
        "cal_size": int(cal_n),
        "qcnt": int(len(test_idx)),
        "k": int(k),
        "efs": int(efs),
        "index_path": index_path,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
