from __future__ import annotations
 
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import median
from typing import Iterator
 
 
@dataclass
class Turn:
 
    stages: dict[str, float] = field(default_factory=dict)
    audio_duration_ms: float | None = None
 
    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())
 
    @property
    def real_time_factor(self) -> float | None:
       
        transcribe = self.stages.get("transcribe")
        if transcribe is None or not self.audio_duration_ms:
            return None
        return transcribe / self.audio_duration_ms
 
 
class Telemetry:
    def __init__(self) -> None:
        self.turns: list[Turn] = []
        self._current: Turn | None = None
 
 
    def begin_turn(self) -> Turn:
        self._current = Turn()
        return self._current
 
    def end_turn(self) -> Turn:
        if self._current is None:
            raise RuntimeError("end_turn() called without begin_turn()")
        turn, self._current = self._current, None
        self.turns.append(turn)
        return turn
 
 
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
       
        if self._current is None:
            raise RuntimeError("stage() called outside a turn")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._current.stages[name] = (
                self._current.stages.get(name, 0.0) + elapsed_ms
            )
 
    def record(self, name: str, value_ms: float) -> None:
        """Attach a timing measured elsewhere (e.g. reported by a library)."""
        if self._current is None:
            raise RuntimeError("record() called outside a turn")
        self._current.stages[name] = value_ms
 
    def set_audio_duration(self, seconds: float) -> None:
        if self._current is None:
            raise RuntimeError("set_audio_duration() called outside a turn")
        self._current.audio_duration_ms = seconds * 1000
 
 
    def report_turn(self, turn: Turn) -> str:
        lines = ["", "  stage            ms      share"]
        lines.append("  " + "-" * 32)
        total = turn.total_ms or 1.0
        for name, ms in turn.stages.items():
            lines.append(f"  {name:<14} {ms:>7.0f}     {ms / total:>5.1%}")
        lines.append("  " + "-" * 32)
        lines.append(f"  {'TOTAL':<14} {turn.total_ms:>7.0f}")
 
        rtf = turn.real_time_factor
        if rtf is not None:
            verdict = "streaming viable" if rtf < 0.5 else "too slow to stream"
            lines.append(f"  real-time factor {rtf:>6.2f}     ({verdict})")
        return "\n".join(lines)
 
    def report_summary(self) -> str:
        if not self.turns:
            return "No turns recorded."
 
        names: list[str] = []
        for turn in self.turns:
            for name in turn.stages:
                if name not in names:
                    names.append(name)
 
        lines = [
            "",
            f"Summary over {len(self.turns)} turns",
            "",
            "  stage             p50       p95       max",
            "  " + "-" * 42,
        ]
        for name in names:
            values = sorted(t.stages[name] for t in self.turns if name in t.stages)
            lines.append(
                f"  {name:<14} {median(values):>7.0f}   "
                f"{_percentile(values, 95):>7.0f}   {values[-1]:>7.0f}"
            )
 
        totals = sorted(t.total_ms for t in self.turns)
        lines.append("  " + "-" * 42)
        lines.append(
            f"  {'TOTAL':<14} {median(totals):>7.0f}   "
            f"{_percentile(totals, 95):>7.0f}   {totals[-1]:>7.0f}"
        )
        return "\n".join(lines)

    def summary_metrics(self) -> dict[str, float]:
        """The same p50/p95/max numbers as report_summary(), as a flat dict
        of floats instead of a formatted string -- for feeding a metrics
        backend (e.g. MLflow) rather than a terminal.
        """
        if not self.turns:
            return {}

        names: list[str] = []
        for turn in self.turns:
            for name in turn.stages:
                if name not in names:
                    names.append(name)

        metrics: dict[str, float] = {}
        for name in names:
            values = sorted(t.stages[name] for t in self.turns if name in t.stages)
            metrics[f"{name}_p50_ms"] = median(values)
            metrics[f"{name}_p95_ms"] = _percentile(values, 95)
            metrics[f"{name}_max_ms"] = values[-1]

        totals = sorted(t.total_ms for t in self.turns)
        metrics["total_p50_ms"] = median(totals)
        metrics["total_p95_ms"] = _percentile(totals, 95)
        metrics["total_max_ms"] = totals[-1]
        return metrics
 
 
def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = math.ceil(pct / 100 * len(sorted_values))
    index = min(len(sorted_values) - 1, max(0, rank - 1))
    return sorted_values[index]
 
