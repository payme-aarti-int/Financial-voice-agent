import time

import pytest

from app.telemetry import Telemetry, Turn, _percentile


def test_stage_records_elapsed_time():
    t = Telemetry()
    t.begin_turn()
    with t.stage("record"):
        time.sleep(0.05)
    turn = t.end_turn()

    assert "record" in turn.stages
    assert turn.stages["record"] >= 50      # at least the sleep
    assert turn.stages["record"] < 200      # generous ceiling for slow CI


def test_repeated_stage_accumulates():
    
    t = Telemetry()
    t.begin_turn()
    with t.stage("llm"):
        time.sleep(0.01)
    with t.stage("llm"):
        time.sleep(0.01)
    turn = t.end_turn()

    assert turn.stages["llm"] >= 20


def test_stage_records_time_even_when_body_raises():
    
    t = Telemetry()
    t.begin_turn()
    with pytest.raises(ValueError):
        with t.stage("transcribe"):
            time.sleep(0.01)
            raise ValueError("boom")
    turn = t.end_turn()

    assert turn.stages["transcribe"] >= 10


def test_total_is_sum_of_stages():
    turn = Turn(stages={"record": 100.0, "transcribe": 50.0})
    assert turn.total_ms == 150.0


def test_real_time_factor():
    """RTF = transcribe time / audio duration. Under 1.0 means the model keeps
    up with live speech, so streaming ASR is viable."""
    turn = Turn(stages={"transcribe": 1000.0}, audio_duration_ms=2000.0)
    assert turn.real_time_factor == 0.5


def test_real_time_factor_is_none_without_audio_duration():
    turn = Turn(stages={"transcribe": 1000.0})
    assert turn.real_time_factor is None


def test_stage_outside_turn_raises():
    """Fail loudly rather than silently dropping a measurement."""
    t = Telemetry()
    with pytest.raises(RuntimeError):
        with t.stage("record"):
            pass


def test_end_turn_without_begin_raises():
    t = Telemetry()
    with pytest.raises(RuntimeError):
        t.end_turn()


def test_summary_across_turns():
    t = Telemetry()
    for _ in range(3):
        t.begin_turn()
        with t.stage("record"):
            pass
        t.end_turn()

    summary = t.report_summary()
    assert "3 turns" in summary
    assert "record" in summary


def test_summary_with_no_turns():
    assert Telemetry().report_summary() == "No turns recorded."


def test_summary_metrics_matches_report_summary():
    t = Telemetry()
    for stage_ms in (100.0, 150.0, 200.0):
        t.begin_turn()
        t.record("agent", stage_ms)
        t.end_turn()

    metrics = t.summary_metrics()
    assert metrics["agent_p50_ms"] == 150.0
    assert metrics["agent_p95_ms"] == 200.0
    assert metrics["agent_max_ms"] == 200.0
    assert metrics["total_p50_ms"] == 150.0
    assert metrics["total_max_ms"] == 200.0


def test_summary_metrics_empty_without_turns():
    assert Telemetry().summary_metrics() == {}


@pytest.mark.parametrize(
    "values,pct,expected",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50, 5),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 10),
        ([42], 95, 42),
    ],
)
def test_percentile(values, pct, expected):
    assert _percentile(values, pct) == expected