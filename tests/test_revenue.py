from app.financial.revenue import RevenueTool


def main():

    revenue_tool = RevenueTool(
        "data/revenue.csv"
    )

    revenue = revenue_tool.get_revenue(
        "2025-10-01",
        "2025-12-31"
    )

    print("\n========== REVENUE ==========")
    print(f"Q4 Revenue: ${revenue:,.2f}")
    print("=============================")


if __name__ == "__main__":
    main()