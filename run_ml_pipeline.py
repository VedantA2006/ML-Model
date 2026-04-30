# ML ENHANCEMENT RUNNER
# Run this to train ML, filter trades, and generate reports
#
# Usage:
#   python run_ml_pipeline.py                 -> full pipeline
#   python run_ml_pipeline.py --train-only    -> train model only
#   python run_ml_pipeline.py --optimize      -> run optimizer too

import subprocess, sys, io, os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

# ── Install ML dependencies ───────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pandas", "numpy", "scikit-learn", "joblib",
    "matplotlib", "requests", "ta",
])

# Optional deps (non-fatal if missing)
for pkg in ["optuna", "shap", "xgboost"]:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print(f"  ⚠️  Optional package '{pkg}' could not be installed (non-fatal)")

import os
import json
import logging
import argparse

# ── Configure logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ml_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("MLRunner")

# ── Resolve paths ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "test2.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "trained_model.pkl")


def run_training():
    """Train the ML model on historical CSV data."""
    from modules.ml_model import MLTradeFilter

    log.info("=" * 70)
    log.info("  PHASE 1: ML MODEL TRAINING")
    log.info(f"  Data source: {CSV_PATH}")
    log.info("=" * 70)

    ml = MLTradeFilter(model_path=MODEL_PATH, csv_path=CSV_PATH)
    results = ml.train()

    print("\n" + "═" * 60)
    print("  ML TRAINING RESULTS")
    print("═" * 60)
    print(f"  Total trades used  : {results['total_trades']}")
    print(f"  Train / Test split : {results['train_size']} / {results['test_size']}")
    print(f"  OOB Score          : {results.get('oob_score', 'N/A')}")
    print("─" * 60)
    print("  TRAIN METRICS:")
    for k, v in results["train_metrics"].items():
        print(f"    {k:<15s}: {v:.4f}")
    print("  FORWARD TEST METRICS:")
    for k, v in results["test_metrics"].items():
        print(f"    {k:<15s}: {v:.4f}")
    print("═" * 60)

    return results


def run_full_pipeline():
    """Run the complete ML enhancement pipeline."""
    from modules.pipeline import MLPipeline

    log.info("=" * 70)
    log.info("  PHASE 2: FULL ML PIPELINE")
    log.info("=" * 70)

    pipeline = MLPipeline(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        output_dir=SCRIPT_DIR,
    )
    comparison = pipeline.run_full_pipeline()

    print("\n" + "═" * 70)
    print("  PERFORMANCE COMPARISON — ORIGINAL vs ML-FILTERED")
    print("═" * 70)
    for key in ["original", "ml_filtered", "high_confidence", "rejected"]:
        stats = comparison.get(key, {})
        if stats:
            label = stats.get("label", key)
            print(f"\n  ── {label} ──")
            print(f"    Trades       : {stats.get('trades', 0)}")
            print(f"    Win Rate     : {stats.get('win_rate', 0)}%")
            print(f"    Profit Factor: {stats.get('profit_factor', 0)}")
            print(f"    Total PnL    : ${stats.get('total_pnl', 0):.2f}")
            print(f"    Max Drawdown : {stats.get('max_drawdown', 0)}%")
    print("═" * 70)

    return comparison


def run_optimizer():
    """Run Optuna parameter optimization (optional)."""
    try:
        import optuna
    except ImportError:
        log.error("Optuna not installed. Run: pip install optuna")
        return None

    from modules.optimizer import AutoOptimizer

    log.info("=" * 70)
    log.info("  PHASE 3: AUTO PARAMETER OPTIMIZATION")
    log.info("=" * 70)

    # We need market data for backtesting — try to import from main strategy
    try:
        sys.path.insert(0, SCRIPT_DIR)
        import test25 as strategy

        log.info("  Fetching market data for optimization backtests...")
        df = strategy.fetch_all_candles(
            strategy.SYMBOL, strategy.INTERVAL, strategy.DURATION
        )
        df = strategy.add_indicators(df)

        optimizer = AutoOptimizer(
            backtest_fn=strategy.backtest,
            add_indicators_fn=strategy.add_indicators,
            n_trials=50,  # reduced for speed; increase for better results
            output_path=os.path.join(SCRIPT_DIR, "optimized_params.json"),
        )
        optimizer.set_data(df)
        results = optimizer.optimize(df)

        print("\n" + "═" * 60)
        print("  OPTIMIZATION RESULTS")
        print("═" * 60)
        print(f"  Best Score : {results.get('best_score', 0):.4f}")
        print(f"  Best Params: {json.dumps(results.get('best_params', {}), indent=4)}")
        print("═" * 60)

        return results

    except Exception as e:
        log.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_shap_analysis():
    """BONUS: SHAP feature analysis (if shap is installed)."""
    try:
        import shap
    except ImportError:
        log.info("  SHAP not installed — skipping SHAP analysis")
        return

    from modules.ml_model import MLTradeFilter, engineer_features

    log.info("  Running SHAP analysis...")

    ml = MLTradeFilter(model_path=MODEL_PATH, csv_path=CSV_PATH)
    ml.load_model()

    raw = ml.load_data(CSV_PATH)
    df = engineer_features(raw)
    valid = df[ml.feature_columns].notna().all(axis=1)
    df_clean = df[valid].reset_index(drop=True)

    X = df_clean[ml.feature_columns]

    # Use TreeExplainer for speed
    explainer = shap.TreeExplainer(ml.model)
    shap_values = explainer.shap_values(X)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[1], X, show=False)
    path = os.path.join(SCRIPT_DIR, "shap_summary.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"  SHAP summary saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML Enhancement Pipeline for Trading Strategy"
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only train the ML model, skip pipeline"
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="Also run Optuna parameter optimization"
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Run SHAP feature importance analysis"
    )
    args = parser.parse_args()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  ML-ENHANCED TRADING SYSTEM — RUNNER".center(68) + "║")
    print("║" + "  Trains from historical CSV → Filters → Reports".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")

    # Verify CSV exists
    if not os.path.exists(CSV_PATH):
        log.error(f"❌ Training data not found: {CSV_PATH}")
        log.error("   Place your historical trade CSV as 'test2.csv' in the same folder")
        sys.exit(1)

    log.info(f"  CSV file found: {CSV_PATH}")

    if args.train_only:
        run_training()
    else:
        run_full_pipeline()

    if args.optimize:
        run_optimizer()

    if args.shap:
        run_shap_analysis()

    # ── Summary of output files ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  OUTPUT FILES")
    print("═" * 60)
    outputs = [
        "trained_model.pkl",
        "filtered_trades.csv",
        "performance_report.json",
        "ml_comparison_chart.png",
        "feature_importance.png",
        "confidence_vs_pnl.png",
        "ml_pipeline.log",
    ]
    if args.optimize:
        outputs.append("optimized_params.json")
    if args.shap:
        outputs.append("shap_summary.png")

    for f in outputs:
        path = os.path.join(SCRIPT_DIR, f)
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists}  {f}")
    print("═" * 60)
    print("\n  🎯 Pipeline complete. Check the output files above.\n")
