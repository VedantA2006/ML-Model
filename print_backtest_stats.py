import pandas as pd
import numpy as np

def print_stats(csv_path, label):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    print("=" * 60)
    print(f"  {label} BACKTEST RESULTS")
    print("=" * 60)

    if df.empty:
        print("  No trades available.")
        return

    # Basic stats
    total = len(df)
    
    # We check if outcome exists, else result
    if "outcome" in df.columns:
        wins = (df["outcome"] == 1).sum()
    elif "result" in df.columns:
        wins = (df["result"] == "TP").sum()
    else:
        wins = 0

    losses = total - wins
    wr = (wins / total) * 100 if total > 0 else 0

    rr = 2.5 # standard RR used in the system
    pf = (wins * rr) / losses if losses > 0 else float("inf")

    # If there are pnl_usd columns, we can use them for real equity calculation
    # Since the original trades in test2.csv had pnl_usd, risk_multiplier etc.
    # Wait, test2.csv has risk_usd, pnl_usd. Filtered trades has risk_multiplier.
    if "pnl_usd" in df.columns:
        if "risk_multiplier" in df.columns:
            df["realised_pnl"] = df["pnl_usd"] * df["risk_multiplier"]
        else:
            df["realised_pnl"] = df["pnl_usd"]
            
        initial_balance = 1000.0
        # Reconstruct balance
        df["new_balance"] = initial_balance + df["realised_pnl"].cumsum()
        final_balance = df["new_balance"].iloc[-1]
    else:
        final_balance = 0

    print(f"  Total Trades       : {total}")
    print(f"  Profitable Trades  : {wins}")
    print(f"  Losing Trades      : {losses}")
    print(f"  Win Rate           : {wr:.2f}%")
    print(f"  Profit Factor      : {pf:.2f}")
    if final_balance > 0:
        print(f"  Final Balance      : ${final_balance:,.2f} (from $1,000)")

    # Monthly / Yearly Returns
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
        df["month"] = df["exit_time"].dt.to_period("M")
        df["year"] = df["exit_time"].dt.to_period("Y")

        if "new_balance" in df.columns:
            # Yearly returns
            yearly_rows = []
            for year, grp in df.groupby("year"):
                grp = grp.sort_values("exit_time")
                idx0 = grp.index[0]
                # Bal before the first trade of the year
                bal0 = grp.loc[idx0, "new_balance"] - grp.loc[idx0, "realised_pnl"]
                if bal0 <= 0: bal0 = 1000.0 # fallback
                pnl_sum = grp["realised_pnl"].sum()
                pct = (pnl_sum / bal0) * 100
                yearly_rows.append({"Year": str(year), "Trades": len(grp), "Return (%)": round(pct, 2)})
            
            print("\n  📆 YEARLY RETURNS:")
            print(pd.DataFrame(yearly_rows).to_string(index=False))

            # Monthly returns
            monthly_rows = []
            for month, grp in df.groupby("month"):
                grp = grp.sort_values("exit_time")
                idx0 = grp.index[0]
                bal0 = grp.loc[idx0, "new_balance"] - grp.loc[idx0, "realised_pnl"]
                if bal0 <= 0: bal0 = 1000.0
                pnl_sum = grp["realised_pnl"].sum()
                pct = (pnl_sum / bal0) * 100
                monthly_rows.append({"Month": str(month), "Trades": len(grp), "Return (%)": round(pct, 2)})

            print("\n  📅 MONTHLY RETURNS:")
            print(pd.DataFrame(monthly_rows).to_string(index=False))

print_stats("test2.csv", "🔴 ORIGINAL (No ML)")
print("\n")
print_stats("filtered_trades.csv", "🟢 ML FILTERED (Prob ≥ 0.40)")
