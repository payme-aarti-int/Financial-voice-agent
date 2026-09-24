"""MLflow observability for the eval harness and the live pipeline.

Tracks latency percentiles (p50/p95/max) and eval pass rates as MLflow
metrics, so you can watch them trend across runs instead of reading a
one-off terminal table.

Local by default -- MLflow writes to ./mlruns unless MLFLOW_TRACKING_URI
points at a server. Set MLFLOW_ENABLED=false to turn logging off
everywhere without touching call sites. Never raises: a tracking backend
being unavailable should not take the agent down with it.
"""

from __future__ import annotations

import logging
import os

import mlflow

from app.telemetry import Telemetry

log = logging.getLogger(__name__)

EVAL_EXPERIMENT = "financial-voice-agent-evals"
PIPELINE_EXPERIMENT = "financial-voice-agent-pipeline"

ENABLED = os.getenv("MLFLOW_ENABLED", "true").strip().lower() not in ("false", "0", "")

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def start_pipeline_run() -> None:
    """One MLflow run per Pipeline lifetime (one CLI invocation, or the
    lifetime of the API process), covering every turn as a logged step.

    Nested if a run is already active (e.g. the eval harness's own run) --
    MLflow only allows one top-level active run at a time.
    """
    if not ENABLED:
        return
    try:
        _configure()
        mlflow.set_experiment(PIPELINE_EXPERIMENT)
        mlflow.start_run(nested=mlflow.active_run() is not None)
    except Exception:
        log.warning("MLflow pipeline run could not start", exc_info=True)


def log_turn(stages: dict[str, float], total_ms: float, step: int) -> None:
    """Log one turn's per-stage timing as a step within the active run."""
    if not ENABLED:
        return
    try:
        metrics = {f"stage_{name}_ms": value for name, value in stages.items()}
        metrics["total_ms"] = total_ms
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        log.warning("MLflow turn logging failed", exc_info=True)


def end_pipeline_run(telemetry: Telemetry) -> None:
    """Log final p50/p95/max across all turns, then close the run."""
    if not ENABLED:
        return
    try:
        metrics = telemetry.summary_metrics()
        if metrics:
            mlflow.log_metrics(metrics)
    except Exception:
        log.warning("MLflow pipeline summary logging failed", exc_info=True)
    finally:
        try:
            mlflow.end_run()
        except Exception:
            log.warning("MLflow run could not be closed cleanly", exc_info=True)


def start_eval_run(category: str | None, threshold: float) -> None:
    """One MLflow run per evals/run.py invocation."""
    if not ENABLED:
        return
    try:
        _configure()
        mlflow.set_experiment(EVAL_EXPERIMENT)
        mlflow.start_run(run_name=category or "all")
        mlflow.log_params({"category": category or "all", "threshold": threshold})
    except Exception:
        log.warning("MLflow eval run could not start", exc_info=True)


def log_eval_summary(
    pass_rate: float,
    latency_p50_ms: float | None,
    latency_p95_ms: float | None,
    latency_max_ms: float | None,
    per_category: dict[str, tuple[int, int]],
) -> None:
    """Log pass rate (overall + per category) and latency percentiles for
    one eval run -- the "p values" you'd want to watch trend over time.
    """
    if not ENABLED:
        return
    try:
        metrics = {"pass_rate": pass_rate}
        if latency_p50_ms is not None:
            metrics["latency_p50_ms"] = latency_p50_ms
            metrics["latency_p95_ms"] = latency_p95_ms
            metrics["latency_max_ms"] = latency_max_ms
        for name, (passed, total) in per_category.items():
            if total:
                metrics[f"pass_rate_{name}"] = passed / total
        mlflow.log_metrics(metrics)
    except Exception:
        log.warning("MLflow eval metric logging failed", exc_info=True)
    finally:
        try:
            mlflow.end_run()
        except Exception:
            log.warning("MLflow eval run could not be closed cleanly", exc_info=True)
