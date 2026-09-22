import pytest

from app.financial.financials import FinancialsRepository

CSV_HEADER = (
    "date,revenue,cogs,salaries,marketing,rnd,admin,"
    "depreciation,amortization,interest,tax\n"
)
CSV_ROWS = [
    "2025-10-15,100000,40000,20000,8000,7000,5000,4000,1500,2000,0\n",
    "2025-11-15,120000,45000,21000,9000,7500,5200,4000,1500,2000,100\n",
    "2025-12-15,140000,50000,22000,9500,8000,5400,4000,1500,2000,200\n",
]


@pytest.fixture
def repository(tmp_path):
    csv_path = tmp_path / "financials.csv"
    csv_path.write_text(CSV_HEADER + "".join(CSV_ROWS))
    return FinancialsRepository(csv_path)


def test_query_revenue_totals_over_a_range(repository):
    result = repository.query(["revenue"], start_month="2025-10", end_month="2025-12")

    assert result["available"] is True
    assert result["months_counted"] == 3
    assert result["values"]["revenue"] == pytest.approx(360000)


def test_query_unknown_metric_is_rejected(repository):
    result = repository.query(["not_a_metric"])
    assert result["available"] is False


def test_query_outside_available_range_reports_range(repository):
    result = repository.query(["revenue"], start_month="2030-01", end_month="2030-12")

    assert result["available"] is False
    assert result["available_range"] == "2025-10 to 2025-12"


def test_rank_periods_orders_by_metric(repository):
    result = repository.rank_periods("revenue", top_n=2)

    assert result["available"] is True
    periods = [r["period"] for r in result["results"]]
    assert periods == ["2025-12", "2025-11"]


def test_compare_periods_reports_change(repository):
    result = repository.compare_periods(
        ["revenue"], "2025-10", "2025-10", "2025-12", "2025-12"
    )

    assert result["available"] is True
    revenue = result["comparison"]["revenue"]
    assert revenue["change"] == pytest.approx(40000)
    assert revenue["direction"] == "up"


def test_get_overview_reports_period(repository):
    overview = repository.get_overview()

    assert overview["available"] is True
    assert overview["period"] == "2025-10 to 2025-12"
    assert overview["months_covered"] == 3
