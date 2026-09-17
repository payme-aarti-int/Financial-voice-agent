from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
 
METRICS = {
    "revenue": "Total sales",
    "cogs": "Cost of goods sold",
    "gross_profit": "Revenue minus COGS",
    "salaries": "Salary expense",
    "marketing": "Marketing expense",
    "rnd": "R&D expense",
    "admin": "General and administrative expense",
    "opex": "Operating expenses: salaries + marketing + R&D + admin",
    "ebitda": "Earnings before interest, tax, depreciation and amortisation",
    "ebit": "EBITDA minus depreciation and amortisation",
    "net_income": "EBIT minus interest and tax",
    "depreciation": "Depreciation",
    "amortization": "Amortisation",
    "interest": "Interest expense",
    "tax": "Tax expense",
    "gross_margin": "Gross profit as a percent of revenue",
    "ebitda_margin": "EBITDA as a percent of revenue",
    "net_margin": "Net income as a percent of revenue",
}
 

RATIO_METRICS = {"gross_margin", "ebitda_margin", "net_margin"}
 
AGGREGATIONS = {"sum", "mean", "min", "max"}
 
 
class Unavailable:
   
 
    def __init__(self, requested: str, first: str, last: str):
        self.requested, self.first, self.last = requested, first, last
 
    def as_dict(self) -> dict:
        return {
            "available": False,
            "requested": self.requested,
            "available_range": f"{self.first} to {self.last}",
        }
 
 
class FinancialsRepository:
    def __init__(self, csv_path: str | Path | None = None):
        path = Path(csv_path or os.getenv("FINANCIALS_CSV_PATH", "data/financials.csv"))
        if not path.exists():
            raise FileNotFoundError(f"Financial data not found at {path}")
 
        frame = pd.read_csv(path)
 
        required = {
            "date", "revenue", "cogs", "salaries", "marketing", "rnd",
            "admin", "depreciation", "amortization", "interest", "tax",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"financials.csv missing columns: {sorted(missing)}")
 
        frame["date"] = pd.to_datetime(frame["date"])
        frame["month"] = frame["date"].dt.strftime("%Y-%m")
        frame["quarter"] = (
            frame["date"].dt.year.astype(str)
            + "-Q"
            + frame["date"].dt.quarter.astype(str)
        )
        frame["year"] = frame["date"].dt.year.astype(str)
 
        # ---- derived lines, defined once, here ----
        frame["gross_profit"] = frame["revenue"] - frame["cogs"]
        frame["opex"] = frame[["salaries", "marketing", "rnd", "admin"]].sum(axis=1)
        frame["ebitda"] = frame["gross_profit"] - frame["opex"]
        frame["ebit"] = frame["ebitda"] - frame["depreciation"] - frame["amortization"]
        frame["net_income"] = frame["ebit"] - frame["interest"] - frame["tax"]
 
        frame["gross_margin"] = (frame["gross_profit"] / frame["revenue"] * 100).round(2)
        frame["ebitda_margin"] = (frame["ebitda"] / frame["revenue"] * 100).round(2)
        frame["net_margin"] = (frame["net_income"] / frame["revenue"] * 100).round(2)
 
        self.frame = frame.sort_values("month").reset_index(drop=True)
 
 
    @property
    def first_month(self) -> str:
        return str(self.frame["month"].iloc[0])
 
    @property
    def last_month(self) -> str:
        return str(self.frame["month"].iloc[-1])
 
    def _unavailable(self, requested: str) -> dict:
        return Unavailable(requested, self.first_month, self.last_month).as_dict()
 
    def _select(self, start_month: str | None, end_month: str | None) -> pd.DataFrame:
        rows = self.frame
        if start_month:
            rows = rows[rows["month"] >= start_month]
        if end_month:
            rows = rows[rows["month"] <= end_month]
        return rows
 
 
    def query(
        self,
        metrics: list[str],
        start_month: str | None = None,
        end_month: str | None = None,
        aggregate: str = "sum",
        group_by: str | None = None,
    ) -> dict:
        
        unknown = [m for m in metrics if m not in METRICS]
        if unknown:
            return {
                "available": False,
                "error": f"Unknown metric(s): {unknown}",
                "valid_metrics": sorted(METRICS),
            }
 
        if aggregate not in AGGREGATIONS:
            return {
                "available": False,
                "error": f"Unknown aggregation '{aggregate}'",
                "valid_aggregations": sorted(AGGREGATIONS),
            }
 
        if group_by not in (None, "month", "quarter", "year"):
            return {
                "available": False,
                "error": f"Cannot group by '{group_by}'",
                "valid_group_by": ["month", "quarter", "year"],
            }
 
        rows = self._select(start_month, end_month)
        if rows.empty:
            return self._unavailable(f"{start_month or 'start'} to {end_month or 'end'}")
 
        effective = {
            m: ("mean" if m in RATIO_METRICS and aggregate == "sum" else aggregate)
            for m in metrics
        }
        notes = [
            f"{m} is a ratio; averaged rather than summed"
            for m, agg in effective.items()
            if agg != aggregate
        ]
 
        result: dict = {
            "available": True,
            "period": f"{rows['month'].iloc[0]} to {rows['month'].iloc[-1]}",
            "months_counted": int(len(rows)),
            "aggregate": aggregate,
        }
        if notes:
            result["notes"] = notes
 
        if group_by:
            grouped: dict[str, dict] = {}
            for key, chunk in rows.groupby(group_by, sort=True):
                grouped[str(key)] = {
                    m: round(float(getattr(chunk[m], effective[m])()), 2)
                    for m in metrics
                }
            result["group_by"] = group_by
            result["groups"] = grouped
        else:
            result["values"] = {
                m: round(float(getattr(rows[m], effective[m])()), 2) for m in metrics
            }
 
        return result
 
 
    def rank_periods(
        self,
        metric: str,
        top_n: int = 3,
        ascending: bool = False,
        group_by: str = "month",
        start_month: str | None = None,
        end_month: str | None = None,
    ) -> dict:
        
        if metric not in METRICS:
            return {
                "available": False,
                "error": f"Unknown metric '{metric}'",
                "valid_metrics": sorted(METRICS),
            }
        if group_by not in ("month", "quarter", "year"):
            return {"available": False, "error": f"Cannot group by '{group_by}'"}
 
        rows = self._select(start_month, end_month)
        if rows.empty:
            return self._unavailable(f"{start_month} to {end_month}")
 
        how = "mean" if metric in RATIO_METRICS else "sum"
        series = getattr(rows.groupby(group_by)[metric], how)()
        ordered = series.sort_values(ascending=ascending).head(top_n)
 
        return {
            "available": True,
            "metric": metric,
            "group_by": group_by,
            "order": "lowest first" if ascending else "highest first",
            "results": [
                {"period": str(k), metric: round(float(v), 2)}
                for k, v in ordered.items()
            ],
        }
 
 
    def compare_periods(
        self,
        metrics: list[str],
        period_a_start: str,
        period_a_end: str,
        period_b_start: str,
        period_b_end: str,
    ) -> dict:
       
        unknown = [m for m in metrics if m not in METRICS]
        if unknown:
            return {
                "available": False,
                "error": f"Unknown metric(s): {unknown}",
                "valid_metrics": sorted(METRICS),
            }
 
        a = self._select(period_a_start, period_a_end)
        b = self._select(period_b_start, period_b_end)
        if a.empty:
            return self._unavailable(f"{period_a_start} to {period_a_end}")
        if b.empty:
            return self._unavailable(f"{period_b_start} to {period_b_end}")
 
        comparison: dict[str, dict] = {}
        for metric in metrics:
            how = "mean" if metric in RATIO_METRICS else "sum"
            a_value = float(getattr(a[metric], how)())
            b_value = float(getattr(b[metric], how)())
            change = b_value - a_value
            comparison[metric] = {
                "period_a": round(a_value, 2),
                "period_b": round(b_value, 2),
                "change": round(change, 2),
                "percent_change": (
                    round(change / abs(a_value) * 100, 2)
                    if a_value and (a_value > 0) == (b_value > 0)
                    else None
                ),
                "direction": "up" if change > 0 else "down" if change < 0 else "flat",
            }
 
        return {
            "available": True,
            "period_a": f"{period_a_start} to {period_a_end}",
            "period_b": f"{period_b_start} to {period_b_end}",
            "comparison": comparison,
        }
 
 
    def get_overview(self) -> dict:
       
        return {
            "available": True,
            "period": f"{self.first_month} to {self.last_month}",
            "months_covered": int(len(self.frame)),
            "available_metrics": METRICS,
            "note": "Balance sheet and cash flow data are not available.",
        }
 
