from __future__ import annotations

"""Core certifier utilities for CRC/LTT pipelines.

This module is intentionally dependency-light and script-friendly so that:
1) calibration can be run from standalone experiment scripts, and
2) trained certifiers can be serialized and reused across evaluations.

Behavior in this file is used by multiple experiment drivers, so changes
should be backwards compatible with the saved JSON/CSV artifacts.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional runtime dependencies (kept optional so basic flows still import).
# NOTE: avoid importing hnswlib at module import time because some experiment
# runners swap in alternate extension builds (sparse L1) in-process.
hnswlib = None

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover
    LogisticRegression = None


@dataclass
class CRCCertifier:
    """Serializable CRC certifier state."""

    k: int
    ef_search: int
    tau: float
    alpha: float
    theta_hat: float
    target_bound: float
    use_augmented: bool = False
    model_kind: str = "margin"
    model_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "ef_search": self.ef_search,
            "tau": self.tau,
            "alpha": self.alpha,
            "theta_hat": self.theta_hat,
            "target_bound": self.target_bound,
            "use_augmented": self.use_augmented,
            "model_kind": self.model_kind,
            "model_params": self.model_params,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CRCCertifier":
        return CRCCertifier(**d)


def _maybe_set_ef(index: Any, ef_search: int) -> None:
    """Set ef-search when the underlying index exposes a setter."""

    if hasattr(index, "set_ef"):
        index.set_ef(int(ef_search))


def default_query_index(index: Any, q: np.ndarray, k: int, ef_search: int) -> Dict[str, Any]:
    """Run one ANN query and collect all telemetry used by certifiers."""

    _maybe_set_ef(index, ef_search)
    set_store = getattr(index, "set_store_candidate_queue_dists", None)
    if callable(set_store):
        try:
            set_store(True)
        except Exception:
            pass
    labels, dists = index.knn_query(np.asarray(q, dtype=np.float32).reshape(1, -1), k=int(k) + 1, num_threads=1)
    labels = np.asarray(labels[0], dtype=np.int64)
    dists = np.asarray(dists[0], dtype=np.float64)

    # Different experiment builds expose different telemetry hooks, so each
    # field below is discovered opportunistically.
    ef_explored = None
    for name in ("get_last_ef_explored", "last_ef_explored", "get_ef_explored"):
        attr = getattr(index, name, None)
        if callable(attr):
            try:
                ef_explored = int(attr())
                break
            except Exception:
                pass
        elif attr is not None:
            try:
                ef_explored = int(attr)
                break
            except Exception:
                pass

    ann_reached = None
    get_last = getattr(index, "get_last_query_computed_dists", None)
    if callable(get_last):
        try:
            last = get_last(False)
            if isinstance(last, (tuple, list)) and len(last) >= 1:
                ids = np.asarray(last[0])
                ann_reached = int(ids.size)
        except Exception:
            pass

    unchecked_dists = np.asarray([], dtype=np.float64)
    get_last_cand = getattr(index, "get_last_candidate_queue_dists", None)
    if callable(get_last_cand):
        try:
            unchecked_dists = np.asarray(get_last_cand(False), dtype=np.float64).reshape(-1)
        except Exception:
            unchecked_dists = np.asarray([], dtype=np.float64)

    best_found_expansion = None
    for name in ("get_last_best_found_expansion", "last_best_found_expansion"):
        attr = getattr(index, name, None)
        if callable(attr):
            try:
                best_found_expansion = int(attr())
                break
            except Exception:
                pass
        elif attr is not None:
            try:
                best_found_expansion = int(attr)
                break
            except Exception:
                pass

    turn_back_count = None
    for name in ("get_last_turn_back_count", "last_turn_back_count"):
        attr = getattr(index, name, None)
        if callable(attr):
            try:
                turn_back_count = int(attr())
                break
            except Exception:
                pass
        elif attr is not None:
            try:
                turn_back_count = int(attr)
                break
            except Exception:
                pass

    turn_back_total_delta = None
    for name in ("get_last_turn_back_total_delta", "last_turn_back_total_delta"):
        attr = getattr(index, name, None)
        if callable(attr):
            try:
                # Keep squared-distance domain to avoid extra sqrt overhead.
                turn_back_total_delta = float(attr(False))
                break
            except Exception:
                try:
                    turn_back_total_delta = float(attr())
                    break
                except Exception:
                    pass
        elif attr is not None:
            try:
                turn_back_total_delta = float(attr)
                break
            except Exception:
                pass

    return {
        "labels_k": labels[:k],
        "dists_k": dists[:k],
        "d1": float(dists[0]),
        "dk": float(dists[k - 1]),
        "dk1": float(dists[k]),
        "ef_explored": ef_explored,
        "ann_reached": ann_reached,
        "ef_search": int(ef_search),
        "unchecked_dists": unchecked_dists,
        "best_found_expansion": best_found_expansion,
        "turn_back_count": turn_back_count,
        "turn_back_total_delta": turn_back_total_delta,
    }


def compute_recall(pred_labels_k: Sequence[int], gt_labels_k: Sequence[int]) -> float:
    """Compute top-k set recall."""

    gt = set(int(x) for x in gt_labels_k)
    if not gt:
        return float("nan")
    pred = set(int(x) for x in pred_labels_k)
    return len(pred.intersection(gt)) / len(gt)


def compute_crc_loss(recall: float, tau: float) -> float:
    """CRC loss: shortfall below the target recall threshold tau."""

    if np.isnan(recall):
        return float("nan")
    return max(0.0, float(tau) - float(recall))


def margin_score(dk: float, dk1: float, eps: float = 1e-12) -> float:
    """Default CRC score based on relative k-th margin."""

    return (float(dk1) - float(dk)) / max(float(dk), eps)


def build_feature_vector(raw: Dict[str, Any], eps: float = 1e-12) -> np.ndarray:
    """Build the augmented feature vector from per-query ANN telemetry."""

    # d1 = float(raw["d1"])
    dk = float(raw["dk"])
    dk1 = float(raw["dk1"])
    topk = np.asarray(raw["dists_k"], dtype=np.float64)
    ef_explored = raw.get("ef_explored", None)
    if ef_explored is None:
        ef_explored = -1.0
    ann_reached = raw.get("ann_reached", None)
    if ann_reached is None:
        ann_reached = -1.0
    ef_search = raw.get("ef_search", None)
    if ef_search is None:
        ef_search = -1.0
    best_found_expansion = raw.get("best_found_expansion", None)
    if best_found_expansion is None:
        best_found_expansion = -1.0
    turn_back_count = raw.get("turn_back_count", None)
    if turn_back_count is None:
        turn_back_count = -1.0
    turn_back_total_delta = raw.get("turn_back_total_delta", None)
    if turn_back_total_delta is None:
        turn_back_total_delta = -1.0

    unchecked = np.asarray(raw.get("unchecked_dists", []), dtype=np.float64).reshape(-1)
    unchecked = np.sort(unchecked)
    efn = int(max(0, int(ef_search))) if float(ef_search) >= 0 else 0
    if efn > 0:
        # Keep a fixed-width tail of unchecked queue distances so saved
        # certifiers remain reusable across evaluation scripts.
        if unchecked.size >= efn:
            unchecked = unchecked[:efn]
        else:
            padv = float(dk1)
            unchecked = np.concatenate([unchecked, np.full((efn - unchecked.size,), padv, dtype=np.float64)])

    margin = (dk1 - dk) / max(dk, eps)
    core = np.asarray(
        [
            margin,
            float(ef_explored),
            float(ann_reached),
            float(ef_search),
            float(best_found_expansion),
            float(turn_back_count),
            float(turn_back_total_delta),
        ],
        dtype=np.float64,
    )
    final_feature = np.concatenate([core, topk, unchecked], axis=0)
    return final_feature


def fit_augmented_score_model(Z: np.ndarray, recalls: np.ndarray, tau: float) -> Tuple[str, Dict[str, Any]]:
    """Fit an augmented scoring model that predicts pass probability."""

    y = (np.asarray(recalls, dtype=np.float64) >= float(tau)).astype(np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    base = np.asarray(Z[:, 0], dtype=np.float64)

    if np.unique(y).size < 2:
        cls = float(y[0]) if y.size else 0.0
        return "sorted_margin_mean", {"x": base.tolist(), "y": [cls for _ in range(len(base))]}

    # Prefer a full-feature model when sklearn is available to improve ranking power.
    if LogisticRegression is not None and Z.shape[0] >= 10:
        mu = np.mean(Z, axis=0)                                                           
        sigma = np.std(Z, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Zn = (Z - mu) / sigma
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(Zn, y.astype(np.int64))
        return "logistic_full", {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }

    if IsotonicRegression is not None:
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(base, y)
        return "isotonic_margin", {"x_thresholds_": iso.X_thresholds_.tolist(), "y_thresholds_": iso.y_thresholds_.tolist()}

    order = np.argsort(base)
    xs = base[order]
    ys = y[order]
    cum = np.cumsum(ys)
    preds = cum / np.arange(1, len(ys) + 1)
    return "sorted_margin_mean", {"x": xs.tolist(), "y": preds.tolist()}


def augmented_score(model_kind: str, model_params: Dict[str, Any], z: np.ndarray) -> float:
    """Score one feature vector with a fitted augmented model."""

    z = np.asarray(z, dtype=np.float64)
    x = float(z[0])
    if model_kind == "logistic_full":
        mu = np.asarray(model_params["mu"], dtype=np.float64)
        sigma = np.asarray(model_params["sigma"], dtype=np.float64)
        coef = np.asarray(model_params["coef"], dtype=np.float64)
        intercept = float(model_params["intercept"])
        zn = (z - mu) / sigma
        logit = float(np.dot(coef, zn) + intercept)
        # Numerically stable sigmoid.
        if logit >= 0:
            ez = math.exp(-logit)
            return float(1.0 / (1.0 + ez))
        ez = math.exp(logit)
        return float(ez / (1.0 + ez))
    if model_kind == "isotonic_margin":
        xs = np.asarray(model_params["x_thresholds_"], dtype=np.float64)
        ys = np.asarray(model_params["y_thresholds_"], dtype=np.float64)
        idx = np.searchsorted(xs, x, side="right") - 1
        idx = int(np.clip(idx, 0, len(ys) - 1))
        return float(ys[idx])
    if model_kind == "sorted_margin_mean":
        xs = np.asarray(model_params["x"], dtype=np.float64)
        ys = np.asarray(model_params["y"], dtype=np.float64)
        idx = np.searchsorted(xs, x, side="right") - 1
        idx = int(np.clip(idx, 0, len(ys) - 1))
        return float(ys[idx])
    raise ValueError(f"Unknown model kind: {model_kind}")


def empirical_crc_risk(scores: np.ndarray, losses: np.ndarray, theta: float) -> float:
    """Compute empirical CRC risk at a fixed acceptance threshold."""

    accepted = scores >= float(theta)
    numer = float(np.nansum(losses[accepted]))
    denom = int(np.sum(accepted)) + 1
    return numer / denom


def calibrate_crc_threshold(rows: List[Dict[str, Any]], alpha: float, tau: float) -> Tuple[float, float]:
    """Return theta_hat and target CRC bound for the given alpha/tau."""

    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    losses = np.asarray([r["loss"] for r in rows], dtype=np.float64)
    n = len(rows)
    target_bound = float(alpha) * (1.0 - float(tau)) * (n / (n + 1.0))
    thetas = np.unique(scores)
    thetas.sort()
    theta_hat = float("inf")
    # Smaller thresholds accept more queries, so take the first threshold that
    # already satisfies the finite-sample CRC risk bound.
    for theta in thetas:
        if empirical_crc_risk(scores, losses, float(theta)) <= target_bound:
            theta_hat = float(theta)
            break
    return theta_hat, target_bound


def build_calibration_table(
    index: Any,
    X_cal: np.ndarray,
    GT_cal: np.ndarray,
    k: int,
    ef_search: int,
    tau: float,
    use_augmented: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, Dict[str, Any]]]]:
    """Evaluate calibration queries and construct per-query calibration rows."""

    rows: List[Dict[str, Any]] = []
    Z: List[np.ndarray] = []
    recalls: List[float] = []

    for i in range(len(X_cal)):
        raw = default_query_index(index, X_cal[i], k, ef_search)
        recall_i = compute_recall(raw["labels_k"], GT_cal[i][:k])
        loss_i = compute_crc_loss(recall_i, tau)
        row: Dict[str, Any] = {
            "qid": int(i),
            "recall": float(recall_i),
            "loss": float(loss_i),
            "labels_k": np.asarray(raw["labels_k"], dtype=np.int64),
            "d1": float(raw["d1"]),
            "dk": float(raw["dk"]),
            "dk1": float(raw["dk1"]),
            "ef_explored": raw.get("ef_explored", None),
        }
        if use_augmented:
            z = build_feature_vector(raw)
            row["z"] = z
            Z.append(z)
            recalls.append(recall_i)
        else:
            row["score"] = margin_score(raw["dk"], raw["dk1"])
        rows.append(row)

    model = None
    if use_augmented:
        # Learn a richer score from ANN telemetry first, then CRC calibrates its
        # acceptance threshold on top of that learned score.
        Zm = np.vstack(Z)
        rm = np.asarray(recalls, dtype=np.float64)
        model = fit_augmented_score_model(Zm, rm, tau)
        kind, params = model
        for row in rows:
            row["score"] = augmented_score(kind, params, row["z"])
    return rows, model


def fit_crc_certifier(
    index: Any,
    X_cal: np.ndarray,
    GT_cal: np.ndarray,
    k: int,
    ef_search: int,
    tau: float,
    alpha: float,
    use_augmented: bool = False,
) -> Tuple[CRCCertifier, List[Dict[str, Any]]]:
    """Fit a CRC certifier and return calibration rows used to fit it."""

    rows, model = build_calibration_table(index, X_cal, GT_cal, k, ef_search, tau, use_augmented)
    theta_hat, target_bound = calibrate_crc_threshold(rows, alpha, tau)
    if model is None:
        kind, params = "margin", None
    else:
        kind, params = model
    certifier = CRCCertifier(
        k=int(k),
        ef_search=int(ef_search),
        tau=float(tau),
        alpha=float(alpha),
        theta_hat=float(theta_hat),
        target_bound=float(target_bound),
        use_augmented=bool(use_augmented),
        model_kind=kind,
        model_params=params,
    )
    # print(rows[0])
    # print(len(rows), "calibration queries, theta_hat", certifier.theta_hat, "target_bound", certifier.target_bound
    #       )
    return certifier, rows


def compute_query_score(index: Any, q: np.ndarray, certifier: CRCCertifier) -> Tuple[float, Dict[str, Any]]:
    """Compute certifier score and raw ANN telemetry for one query."""

    raw = default_query_index(index, q, certifier.k, certifier.ef_search)
    if certifier.use_augmented:
        # Rebuild the same augmented score used during calibration directly from
        # the live ANN telemetry of this query.
        z = build_feature_vector(raw)
        score = augmented_score(certifier.model_kind, certifier.model_params or {}, z)
    else:
        score = margin_score(raw["dk"], raw["dk1"])
    return float(score), raw


def crc_decision(index: Any, q: np.ndarray, certifier: CRCCertifier) -> Dict[str, Any]:
    """Return accept/escalate decision and ANN top-k output for one query."""

    score, raw = compute_query_score(index, q, certifier)
    accepted = bool(score >= certifier.theta_hat)
    return {
        "accepted": accepted,
        "score": float(score),
        "labels": np.asarray(raw["labels_k"], dtype=np.int64),
        "dists": np.asarray(raw["dists_k"], dtype=np.float64),
    }


def evaluate_crc(index: Any, X_test: np.ndarray, GT_test: np.ndarray, certifier: CRCCertifier) -> Dict[str, Any]:
    """Evaluate certifier behavior over a held-out query set."""

    weighted_losses: List[float] = []
    accepted_losses: List[float] = []
    accepted_recalls: List[float] = []
    dangerous = 0
    accepted = 0

    for i in range(len(X_test)):
        out = crc_decision(index, X_test[i], certifier)
        recall = compute_recall(out["labels"], GT_test[i][: certifier.k])
        loss = compute_crc_loss(recall, certifier.tau)
        if out["accepted"]:
            accepted += 1
            accepted_losses.append(loss)
            accepted_recalls.append(recall)
            weighted_losses.append(loss)
            if recall < certifier.tau:
                dangerous += 1
        else:
            weighted_losses.append(0.0)

    n = len(X_test)
    return {
        "n_test": int(n),
        "acceptance_rate": float(accepted / n) if n else float("nan"),
        "empirical_crc_risk": float(np.mean(np.asarray(weighted_losses, dtype=np.float64))) if n else float("nan"),
        "mean_shortfall_accepted": float(np.mean(np.asarray(accepted_losses, dtype=np.float64))) if accepted_losses else 0.0,
        "mean_recall_accepted": float(np.mean(np.asarray(accepted_recalls, dtype=np.float64))) if accepted_recalls else float("nan"),
        "dangerous_rate_among_accepted": float(dangerous / accepted) if accepted else 0.0,
        "num_accept": int(accepted),
        "num_escalate": int(n - accepted),
    }


def save_certifier(certifier: CRCCertifier, path: str | Path) -> None:
    """Serialize certifier JSON for later replay."""

    Path(path).write_text(json.dumps(certifier.to_dict(), indent=2))


def load_certifier(path: str | Path) -> CRCCertifier:
    """Load certifier JSON produced by save_certifier."""

    return CRCCertifier.from_dict(json.loads(Path(path).read_text()))


def save_calibration_rows(rows: List[Dict[str, Any]], path: str | Path) -> None:
    """Write calibration table with stable columns for analysis scripts."""

    fieldnames = ["qid", "score", "recall", "loss", "d1", "dk", "dk1", "ef_explored"]
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            out = {k: row.get(k, None) for k in fieldnames}
            w.writerow(out)
