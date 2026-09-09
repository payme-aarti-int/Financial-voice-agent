from __future__ import annotations
 
import os
from dataclasses import dataclass
from pathlib import Path
 
import pandas as pd
 
 
@dataclass(frozen=True)
class Unavailable:
 
    requested: str
    available_from: str
    available_to: str
 
    def as_dict(self) -> dict:
        return {
            "available": False,
            "requested": self.requested,
            "available_range": f"{self.available_from} to {self.available_to}",
        }
 
 
class RevenueRepository:
    def __init__(self, csv_path: str | Path | None = None):
        path = Path(csv_path or os.getenv("REVENUE_CSV_PATH", "data/revenue.csv"))
        if not path.exists():
            raise FileNotFoundError(f"Revenue data not found at {path}")
 
        frame = pd.read_csv(path)
 
        missing = {"date", "amount"} - set(frame.columns)
        if missing:
            raise ValueError(f"revenue.csv missing required columns: {missing}")
 
        frame["date"] = pd.to_datetime(frame["date"])
        frame["amount"] = pd.to_numeric(frame["amount"])
 
        
        frame["month"] = frame["date"].dt.strftime("%Y-%m")
        self.frame = frame.sort_values("month").reset_index(drop=True)
 
 
    @property
    def first_month(self) -> str:
        return str(self.frame["month"].iloc[0])
 
    @property
    def last_month(self) -> str:
        return str(self.frame["month"].iloc[-1])
 
    def _unavailable(self, requested: str) -> Unavailable:
        return Unavailable(requested, self.first_month, self.last_month)
 
 
    def get_revenue_range(self, start_month: str, end_month: str | None = None) -> dict:
        
        end_month = end_month or start_month
        rows = self.frame[
            (self.frame["month"] >= start_month) & (self.frame["month"] <= end_month)
        ]
 
        if rows.empty:
            return self._unavailable(f"{start_month} to {end_month}").as_dict()
 
        return {
            "available": True,
            "start_month": start_month,
            "end_month": end_month,
            "months_counted": int(len(rows)),
            "total": float(rows["amount"].sum()),
            "average_per_month": round(float(rows["amount"].mean()), 2),
            "by_month": {
                str(m): float(a) for m, a in zip(rows["month"], rows["amount"])
            },
        }
 
 
    def get_growth(self, from_month: str, to_month: str) -> dict:
        start = self.frame[self.frame["month"] == from_month]
        end = self.frame[self.frame["month"] == to_month]
 
        if start.empty:
            return self._unavailable(from_month).as_dict()
        if end.empty:
            return self._unavailable(to_month).as_dict()
 
        start_amount = float(start["amount"].iloc[0])
        end_amount = float(end["amount"].iloc[0])
        change = end_amount - start_amount
 
        return {
            "available": True,
            "from_month": from_month,
            "to_month": to_month,
            "from_amount": start_amount,
            "to_amount": end_amount,
            "change": change,
            "percent_change": (
                round(change / start_amount * 100, 2) if start_amount else None
            ),
            "direction": "up" if change > 0 else "down" if change < 0 else "flat",
        }
 
 
    def get_summary(self) -> dict:
        """Totals, extremes, and trend across all available data."""
        amounts = self.frame["amount"]
        best = self.frame.loc[amounts.idxmax()]
        worst = self.frame.loc[amounts.idxmin()]
 
        first_amount = float(amounts.iloc[0])
        last_amount = float(amounts.iloc[-1])
 
        return {
            "available": True,
            "months_covered": int(len(self.frame)),
            "period": f"{self.first_month} to {self.last_month}",
            "total": float(amounts.sum()),
            "average_per_month": round(float(amounts.mean()), 2),
            "best_month": {"month": str(best["month"]), "amount": float(best["amount"])},
            "worst_month": {
                "month": str(worst["month"]),
                "amount": float(worst["amount"]),
            },
            "trend": (
                "growing"
                if last_amount > first_amount
                else "declining"
                if last_amount < first_amount
                else "flat"
            ),
            "overall_percent_change": (
                round((last_amount - first_amount) / first_amount * 100, 2)
                if first_amount
                else None
            ),
        }
 
