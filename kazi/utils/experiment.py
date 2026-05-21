"""
Experiment tracking for kazi.

Logs per-run metrics (tokens, cost, latency) to MLflow or Weights & Biases.
Silently skips when neither package is installed — no import error, no overhead.

Install::

    pip install mlflow                  # MLflow
    pip install wandb                   # Weights & Biases

Quick start::

    from kazi.utils.experiment import ExperimentTracker

    tracker = ExperimentTracker(backend="mlflow", project="my-agent")
    # or
    tracker = ExperimentTracker(backend="wandb", project="my-agent")

    async with await Kazi.create(config) as kazi:
        reply = await kazi.run("Summarise Q3 results", track_cost=True)
        tracker.log_run(
            message="Summarise Q3 results",
            reply=reply.reply,
            model=config.llm.model,
            cost=reply.cost,
            latency_ms=elapsed_ms,
            thread_id="session-1",
        )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ExperimentTracker:
    """
    Thin wrapper around MLflow or W&B that logs kazi.run() metrics.

    Both backends are optional. When the library isn't installed, all
    log_run() calls are silent no-ops.

    Parameters
    ----------
    backend     "mlflow" or "wandb"
    project     Experiment / project name (MLflow experiment or W&B project)
    run_name    Optional display name for individual runs
    """

    def __init__(
        self,
        backend: str = "mlflow",
        *,
        project: str | None = "kazi",
        run_name: str | None = None,
        tags: dict | None = None,
    ) -> None:
        if backend not in ("mlflow", "wandb"):
            raise ValueError(f"backend must be 'mlflow' or 'wandb', got {backend!r}")
        self.backend = backend
        self.project = project
        self.run_name = run_name
        self.tags = tags or {}
        self._client = None
        self._init()

    def _init(self) -> None:
        if self.backend == "mlflow":
            try:
                import mlflow
                if self.project:
                    mlflow.set_experiment(self.project)
                self._client = mlflow
                logger.debug("ExperimentTracker: MLflow backend ready (project=%s)", self.project)
            except ImportError:
                logger.debug("mlflow not installed — experiment tracking disabled")
        else:
            try:
                import wandb
                self._client = wandb
                logger.debug("ExperimentTracker: W&B backend ready (project=%s)", self.project)
            except ImportError:
                logger.debug("wandb not installed — experiment tracking disabled")

    def log_run(
        self,
        *,
        message: str,
        reply: str,
        model: str,
        thread_id: str = "default",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        extra: dict | None = None,
    ) -> None:
        """
        Log a completed kazi.run() call.

        Accepts either raw token counts or a RunCost object via the convenience
        wrapper log_run_result().
        """
        if self._client is None:
            return

        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        params = {"model": model, "thread_id": thread_id, **self.tags}
        if extra:
            params.update(extra)

        try:
            if self.backend == "mlflow":
                with self._client.start_run(run_name=self.run_name, nested=True):
                    self._client.log_params(params)
                    self._client.log_metrics(metrics)
                    self._client.log_text(message[:10_000], "input.txt")
                    self._client.log_text(reply[:10_000], "output.txt")
            else:
                if not self._client.run:
                    self._client.init(
                        project=self.project,
                        name=self.run_name,
                        tags=list(self.tags.keys()),
                    )
                self._client.log({**metrics, **params})
        except Exception as exc:
            logger.warning("ExperimentTracker: failed to log run: %s", exc)

    def log_run_result(
        self,
        *,
        message: str,
        result,
        model: str,
        thread_id: str = "default",
        latency_ms: float = 0.0,
        extra: dict | None = None,
    ) -> None:
        """
        Convenience wrapper: pass the RunResult returned by kazi.run(track_cost=True).
        Extracts token counts and cost automatically.
        """
        from kazi.core.cost import RunResult

        if not isinstance(result, RunResult):
            self.log_run(
                message=message,
                reply=str(result),
                model=model,
                thread_id=thread_id,
                latency_ms=latency_ms,
                extra=extra,
            )
            return

        self.log_run(
            message=message,
            reply=result.reply if isinstance(result.reply, str) else str(result.reply),
            model=model,
            thread_id=thread_id,
            input_tokens=result.cost.input_tokens,
            output_tokens=result.cost.output_tokens,
            cost_usd=result.cost.cost_usd,
            latency_ms=latency_ms,
            extra=extra,
        )

    def finish(self) -> None:
        """Call at shutdown to flush W&B runs. No-op for MLflow."""
        if self._client is None:
            return
        if self.backend == "wandb":
            try:
                if self._client.run:
                    self._client.finish()
            except Exception:
                pass
