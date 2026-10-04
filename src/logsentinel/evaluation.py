"""Train, tune and compare detectors.

Protocol (the same for every model, so the comparison is fair):

1. Fit the featurizer and the detector on a **healthy training period**
   (in production you would pick a period without known incidents).
2. Choose each detector's alert threshold by maximising F1 on a separate
   **validation period** that contains incidents.
3. Report metrics on an unseen **test period**.

Besides window-level precision/recall/F1 we report metrics an on-call
engineer cares about: how many incidents were caught, how quickly, and how
many false alarms per day the model would page you with.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import synthetic
from .bundle import ModelBundle
from .features import Window, WindowFeaturizer, count_windows, label_by_intervals, time_windows
from .models import build_detector
from .parsing import parse_lines

PRETTY = {
    "static_threshold": "Static error-rate threshold",
    "isolation_forest": "Isolation Forest",
    "lstm_autoencoder": "LSTM Autoencoder",
}


# -- metrics -------------------------------------------------------------------------
def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    if labels.sum() == 0:
        return float(np.quantile(scores, 0.99))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def window_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = 0 < labels.sum() < len(labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else float("nan"),
    }


def incident_metrics(windows: Sequence[Window], predictions: np.ndarray, incidents: Sequence) -> dict:
    """Incident recall, detection delay and false-alarm episodes per day."""
    detected, delays, by_kind = 0, [], {}
    for incident in incidents:
        hits = [w for w, p in zip(windows, predictions) if p and incident.start < w.end and w.start < incident.end]
        caught = bool(hits)
        detected += caught
        stats = by_kind.setdefault(incident.kind, [0, 0])
        stats[0] += caught
        stats[1] += 1
        if caught:
            delays.append(max((hits[0].end - incident.start).total_seconds() / 60, 0.0))

    false_episodes, in_episode = 0, False
    for window, pred in zip(windows, predictions):
        is_false = bool(pred) and window.label == 0
        if is_false and not in_episode:
            false_episodes += 1
        in_episode = is_false
    days = max((windows[-1].end - windows[0].start).total_seconds() / 86400, 1e-9) if windows else 1.0
    return {
        "incident_recall": detected / len(incidents) if incidents else float("nan"),
        "incidents_detected": f"{detected}/{len(incidents)}",
        "mean_detection_delay_min": float(np.mean(delays)) if delays else float("nan"),
        "false_alarms_per_day": false_episodes / days,
        "recall_by_type": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())},
    }


def apply_novelty_rule(featurizer: WindowFeaturizer, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Always alert when an ERROR-level message appears that was never seen in training."""
    idx = featurizer.feature_names.index("log_unseen_error_templates")
    return np.where(X[:, idx] > 0, 1, predictions)


# -- experiments ---------------------------------------------------------------------
@dataclass
class ExperimentResult:
    dataset: str
    rows: list[dict] = field(default_factory=list)
    bundles: dict[str, ModelBundle] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        has_incidents = any("incident_recall" in r for r in self.rows)
        header = "| Model | Precision | Recall | F1 | ROC-AUC |"
        sep = "|---|---:|---:|---:|---:|"
        if has_incidents:
            header += " Incidents caught | Avg. delay (min) | False alarms/day |"
            sep += "---:|---:|---:|"
        lines = [header, sep]
        for r in self.rows:
            line = f"| {r['model']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | {r['roc_auc']:.3f} |"
            if has_incidents:
                line += f" {r['incidents_detected']} | {r['mean_detection_delay_min']:.1f} | {r['false_alarms_per_day']:.1f} |"
            lines.append(line)
        return "\n".join(lines)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"results_{self.dataset}.json").write_text(
            json.dumps({"info": self.info, "rows": self.rows}, indent=2, default=str)
        )
        (directory / f"results_{self.dataset}.md").write_text(self.to_markdown() + "\n")


def _evaluate(
    name: str,
    featurizer: WindowFeaturizer,
    train: list[Window],
    val: list[Window],
    test: list[Window],
    window_seconds: int,
    incidents_test: Sequence = (),
    novelty_variants: bool = True,
    detector_kwargs: dict | None = None,
) -> tuple[list[dict], ModelBundle]:
    X_train, X_val, X_test = (featurizer.transform(w) for w in (train, val, test))
    y_val = np.array([w.label for w in val])
    y_test = np.array([w.label for w in test])

    started = time.perf_counter()
    detector = build_detector(name, **(detector_kwargs or {})).fit(X_train, featurizer.feature_names)
    fit_seconds = time.perf_counter() - started
    threshold = best_threshold(detector.score(X_val), y_val)
    scores = detector.score(X_test)
    predictions = (scores >= threshold).astype(int)

    variants = [("", predictions)]
    if novelty_variants and name != "static_threshold":
        variants.append((" + novelty rule", apply_novelty_rule(featurizer, X_test, predictions)))
    rows = []
    for suffix, preds in variants:
        row = {"model": PRETTY.get(name, name) + suffix, "detector": name, "threshold": threshold,
               "fit_seconds": round(fit_seconds, 2), **window_metrics(y_test, scores, preds)}
        if incidents_test:
            row.update(incident_metrics(test, preds, incidents_test))
        rows.append(row)
    bundle = ModelBundle(featurizer, detector, threshold, window_seconds, {"trained_on": "", "detector": name})
    return rows, bundle


def run_synthetic_experiment(
    detectors: Sequence[str] = ("static_threshold", "isolation_forest", "lstm_autoencoder"),
    train_hours: float = 48,
    eval_hours: float = 24,
    window_seconds: int = 60,
    seed: int = 7,
    lines_per_second: float = 2.0,
    detector_kwargs: dict[str, dict] | None = None,
    log=print,
) -> ExperimentResult:
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    log(f"Generating {train_hours:.0f}h healthy training period ...")
    train_ds = synthetic.generate(start, train_hours, lines_per_second, incidents_per_day=0, seed=seed)
    val_start = train_ds.end
    log(f"Generating {eval_hours:.0f}h validation and {eval_hours:.0f}h test periods with incidents ...")
    val_ds = synthetic.generate(val_start, eval_hours, lines_per_second, incidents_per_day=12, seed=seed + 1)
    test_ds = synthetic.generate(val_ds.end, eval_hours, lines_per_second, incidents_per_day=12, seed=seed + 2)

    splits = []
    for ds in (train_ds, val_ds, test_ds):
        windows = time_windows(ds.records, window_seconds, ds.start, ds.end)
        label_by_intervals(windows, ds.incidents)
        splits.append(windows)
    train, val, test = splits
    featurizer = WindowFeaturizer().fit(train)

    result = ExperimentResult(
        dataset="synthetic",
        info={
            "log_lines": sum(len(d.records) for d in (train_ds, val_ds, test_ds)),
            "windows": {"train": len(train), "val": len(val), "test": len(test)},
            "anomalous_test_windows": int(sum(w.label for w in test)),
            "test_incidents": len(test_ds.incidents),
            "window_seconds": window_seconds,
            "features": len(featurizer.feature_names),
            "seed": seed,
        },
    )
    for name in detectors:
        log(f"Training and evaluating {PRETTY.get(name, name)} ...")
        rows, bundle = _evaluate(name, featurizer, train, val, test, window_seconds, test_ds.incidents,
                                 detector_kwargs=(detector_kwargs or {}).get(name))
        bundle.metadata.update({"trained_on": "synthetic", "test_metrics": rows[-1]})
        result.rows.extend(rows)
        result.bundles[name] = bundle
    return result


def run_bgl_experiment(
    path: str | Path,
    detectors: Sequence[str] = ("static_threshold", "isolation_forest", "lstm_autoencoder"),
    window_size: int = 20,
    step: int = 10,
    max_lines: int | None = None,
    detector_kwargs: dict[str, dict] | None = None,
    log=print,
) -> ExperimentResult:
    """Evaluate on BlueGene/L logs (Loghub). Uses fixed-size windows of log lines.

    Chronological split: first 50% train (normal windows only), next 20%
    validation, last 30% test.
    """
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines() if max_lines is None else [next(fh, "") for _ in range(max_lines)]
    records = parse_lines(lines, fmt="bgl")
    log(f"Parsed {len(records):,} BGL records")
    windows = count_windows(records, window_size, step)
    n = len(windows)
    train = [w for w in windows[: int(n * 0.5)] if w.label == 0]
    val, test = windows[int(n * 0.5) : int(n * 0.7)], windows[int(n * 0.7) :]
    featurizer = WindowFeaturizer(min_count=1).fit(train)
    result = ExperimentResult(
        dataset="bgl",
        info={
            "log_lines": len(records),
            "anomalous_lines": int(sum(r.label for r in records)),
            "windows": {"train_normal": len(train), "val": len(val), "test": len(test)},
            "anomalous_test_windows": int(sum(w.label for w in test)),
            "window": f"{window_size} lines, step {step}",
        },
    )
    for name in detectors:
        log(f"Training and evaluating {PRETTY.get(name, name)} ...")
        kwargs = dict((detector_kwargs or {}).get(name) or {})
        if name == "lstm_autoencoder":
            kwargs.setdefault("seq_len", 3)
        rows, bundle = _evaluate(name, featurizer, train, val, test, 0, novelty_variants=True, detector_kwargs=kwargs)
        bundle.metadata.update({"trained_on": "bgl", "test_metrics": rows[-1]})
        result.rows.extend(rows)
        result.bundles[name] = bundle
    return result


def train_production_bundle(
    detector: str = "isolation_forest",
    hours: float = 48,
    window_seconds: int = 60,
    seed: int = 7,
    lines_per_second: float = 2.0,
) -> ModelBundle:
    """Train the model the streaming service uses, with a threshold tuned on validation data."""
    train_ds = synthetic.generate(datetime(2026, 1, 5, tzinfo=timezone.utc), hours, lines_per_second, 0, seed)
    val_ds = synthetic.generate(train_ds.end, 24, lines_per_second, 12, seed + 1)
    train = time_windows(train_ds.records, window_seconds, train_ds.start, train_ds.end)
    val = time_windows(val_ds.records, window_seconds, val_ds.start, val_ds.end)
    label_by_intervals(val, val_ds.incidents)
    featurizer = WindowFeaturizer().fit(train)
    X_train = featurizer.transform(train)
    model = build_detector(detector).fit(X_train, featurizer.feature_names)
    y_val = np.array([w.label for w in val])
    threshold = best_threshold(model.score(featurizer.transform(val)), y_val)
    return ModelBundle(
        featurizer, model, threshold, window_seconds,
        {"detector": detector, "trained_on": "synthetic", "train_hours": hours,
         "trained_at": datetime.now(timezone.utc).isoformat(), "train_windows": len(train)},
    )

