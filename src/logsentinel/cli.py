"""Command-line entry point: ``logsentinel <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import get_settings

logger = logging.getLogger("logsentinel")


def _default_detector() -> str:
    try:
        import torch  # noqa: F401

        return "lstm_autoencoder"
    except ImportError:
        return "isolation_forest"


# -- commands --------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    from . import synthetic

    start = datetime.now(timezone.utc) - timedelta(hours=args.hours) if args.ending_now else None
    ds = synthetic.generate(start, args.hours, args.rate, args.incidents_per_day, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ds.lines()) + "\n")
    truth = [{"kind": i.kind, "start": i.start.isoformat(), "end": i.end.isoformat(), "intensity": round(i.intensity, 2)}
             for i in ds.incidents]
    out.with_suffix(".incidents.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {len(ds.records):,} lines and {len(ds.incidents)} ground-truth incidents to {out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluation import run_bgl_experiment, run_synthetic_experiment

    detectors = args.detectors or ["static_threshold", "isolation_forest", "lstm_autoencoder"]
    if args.dataset == "synthetic":
        result = run_synthetic_experiment(detectors, train_hours=args.train_hours, seed=args.seed)
    else:
        if not args.path:
            sys.exit("--path is required for the bgl dataset")
        result = run_bgl_experiment(args.path, detectors, window_size=args.window_size, step=args.step,
                                    max_lines=args.max_lines)
    result.save(args.reports_dir)
    print()
    print(json.dumps(result.info, indent=2))
    print()
    print(result.to_markdown())
    if args.dataset == "synthetic":
        print("\nIncidents caught by type:")
        for row in result.rows:
            print(f"  {row['model']:<34} {row['recall_by_type']}")
    if args.models_dir:
        for name, bundle in result.bundles.items():
            bundle.save(Path(args.models_dir) / f"{args.dataset}_{name}.pkl")
    print(f"\nReports written to {args.reports_dir}/")


def cmd_train(args: argparse.Namespace) -> None:
    from .evaluation import train_production_bundle

    out = Path(args.out)
    if args.skip_if_exists and out.exists():
        print(f"Model already exists at {out}, skipping training")
        return
    detector = args.detector or _default_detector()
    print(f"Training {detector} on {args.hours:.0f}h of healthy synthetic logs ...")
    bundle = train_production_bundle(detector, args.hours, args.window_seconds, args.seed)
    bundle.save(out)
    print(f"Saved model to {out} (threshold={bundle.threshold:.4f})")


def cmd_replay(args: argparse.Namespace) -> None:
    from .bundle import ModelBundle
    from .pipeline import Pipeline
    from .storage import make_repository
    from .summarizer import Summarizer

    settings = get_settings()
    bundle = ModelBundle.load(args.model or settings.model_path)
    repo = make_repository(args.database_url or settings.database_url)
    summarizer = Summarizer(settings.anthropic_api_key, settings.llm_model)
    pipeline = Pipeline(bundle, repo, summarizer, log_format=args.format)
    with open(args.file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            pipeline.handle_line(line)
    pipeline.flush()
    incidents = repo.recent_incidents(limit=1000)[::-1]
    print(f"\nDetected {len(incidents)} incident(s):\n")
    for inc in incidents:
        print(f"#{inc['id']:<3} {inc['started_at'][:16]} -> {inc['ended_at'][11:16]}  "
              f"({inc['windows']} windows, peak score {inc['peak_score']:.3f})")
        print(f"     {inc['summary']}\n")


def cmd_produce(args: argparse.Namespace) -> None:
    from . import synthetic
    from .kafka_io import produce

    settings = get_settings()
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    else:
        start = datetime.now(timezone.utc) - timedelta(hours=args.synthetic_hours)
        ds = synthetic.generate(start, args.synthetic_hours, incidents_per_day=args.incidents_per_day, seed=args.seed)
        lines = ds.lines()
        print(f"Generated {len(lines):,} synthetic lines with {len(ds.incidents)} incidents:")
        for inc in ds.incidents:
            print(f"  {inc.kind:<22} {inc.start:%H:%M} - {inc.end:%H:%M} UTC  intensity {inc.intensity:.2f}")
    sent = produce(lines, args.bootstrap or settings.kafka_bootstrap, args.topic or settings.kafka_topic,
                   args.speed, args.format)
    print(f"Produced {sent:,} lines")


def cmd_consume(args: argparse.Namespace) -> None:
    from prometheus_client import start_http_server

    from .bundle import ModelBundle
    from .kafka_io import consume
    from .pipeline import Pipeline
    from .storage import make_repository
    from .summarizer import Summarizer

    settings = get_settings()
    model_path = Path(settings.model_path)
    while not model_path.exists():
        logger.info("Waiting for model at %s ...", model_path)
        time.sleep(5)
    bundle = ModelBundle.load(model_path)
    start_http_server(settings.metrics_port)
    summarizer = Summarizer(settings.anthropic_api_key, settings.llm_model)
    logger.info("Loaded %s (threshold %.4f); LLM summaries: %s", bundle.detector.name, bundle.threshold,
                "on" if summarizer.uses_llm else "off")
    pipeline = Pipeline(bundle, make_repository(settings.database_url), summarizer, settings.log_format)
    consume(pipeline, settings.kafka_bootstrap, settings.kafka_topic, settings.kafka_group)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


# -- parser ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="logsentinel", description="Real-time log anomaly detection")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Generate a labelled synthetic log file")
    p.add_argument("--out", default="data/synthetic.log")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--rate", type=float, default=2.0, help="average lines per second")
    p.add_argument("--incidents-per-day", type=float, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ending-now", action="store_true", help="make the log end at the current time")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="Benchmark detectors and write a results report")
    p.add_argument("--dataset", choices=["synthetic", "bgl"], default="synthetic")
    p.add_argument("--path", help="path to BGL.log (Loghub) for --dataset bgl")
    p.add_argument("--detectors", nargs="+", choices=["static_threshold", "isolation_forest", "lstm_autoencoder"])
    p.add_argument("--train-hours", type=float, default=48)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window-size", type=int, default=20, help="BGL: lines per window")
    p.add_argument("--step", type=int, default=10, help="BGL: lines between window starts")
    p.add_argument("--max-lines", type=int, help="BGL: only read the first N lines")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--models-dir", help="also save every trained model here")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="Train the model used by the streaming service")
    p.add_argument("--detector", choices=["static_threshold", "isolation_forest", "lstm_autoencoder"])
    p.add_argument("--out", default="models/model.pkl")
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--window-seconds", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-if-exists", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("replay", help="Run the full pipeline over a log file without Kafka")
    p.add_argument("--file", required=True)
    p.add_argument("--format", choices=["app", "bgl"], default="app")
    p.add_argument("--model")
    p.add_argument("--database-url")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("produce", help="Replay logs into Kafka")
    p.add_argument("--file", help="log file to replay; omit to stream synthetic logs")
    p.add_argument("--synthetic-hours", type=float, default=6)
    p.add_argument("--incidents-per-day", type=float, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--speed", type=float, default=30, help="replay speed-up factor (0 = as fast as possible)")
    p.add_argument("--format", choices=["app", "bgl"], default="app")
    p.add_argument("--bootstrap")
    p.add_argument("--topic")
    p.set_defaults(func=cmd_produce)

    p = sub.add_parser("consume", help="Run the streaming detector on Kafka (configured via env vars)")
    p.set_defaults(func=cmd_consume)

    p = sub.add_parser("serve", help="Run the REST API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
