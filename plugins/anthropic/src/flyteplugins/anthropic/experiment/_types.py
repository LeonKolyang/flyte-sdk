"""Shared types for the ML experiment framework."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

Goal = Literal["minimize", "maximize"]


@dataclass
class TrialRecord:
    """Record of a single pipeline trial run by the agent."""

    iteration: int
    task_name: str
    metric: float | None
    improved: bool
    error: str | None = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "task_name": self.task_name,
            "metric": self.metric,
            "improved": self.improved,
            "error": self.error,
            "summary": self.summary,
        }


@dataclass
class ExperimentHistory:
    """Accumulates trial records and tracks the best metric seen so far."""

    goal: Goal
    records: list[TrialRecord] = field(default_factory=list)
    best_metric: float | None = None

    def add(self, record: TrialRecord) -> None:
        self.records.append(record)
        if record.metric is not None:
            if self.best_metric is None or self._is_better(record.metric, self.best_metric):
                self.best_metric = record.metric
                record.improved = True

    def _is_better(self, candidate: float, current_best: float) -> bool:
        return candidate < current_best if self.goal == "minimize" else candidate > current_best

    def to_json(self) -> str:
        return json.dumps(
            {
                "goal": self.goal,
                "best_metric": self.best_metric,
                "total_trials": len(self.records),
                "trials": [r.to_dict() for r in self.records],
            },
            indent=2,
        )
