import csv
from datetime import datetime


class RevenueTool:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_revenue(self, start_date: str, end_date: str) -> float:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        total = 0.0

        with open(self.data_path, "r") as file:
            reader = csv.DictReader(file)

            for row in reader:
                date = datetime.strptime(
                    row["date"],
                    "%Y-%m-%d"
                ).date()

                if start <= date <= end:
                    total += float(row["amount"])

        return total