# ╔══════════════════════════════════════════════════════════════╗
# ║  ML PIPELINE ORCHESTRATOR                                   ║
# ║  End-to-end: Train → Filter → Size → Report                 ║
# ╚══════════════════════════════════════════════════════════════╝

import json, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional

from .ml_model import MLTradeFilter, engineer_features
from .position_sizing import PositionSizer
from .exit_model import ExitOptimizer

log = logging.getLogger("MLPipeline")


class MLPipeline:
    """
    Full ML enhancement pipeline that wraps the existing strategy.
    Orchestrates: training, filtering, sizing, exit optimization, reporting.
    """

    def __init__(
        self,
        csv_path: str = "test2.csv",
        model_path: str = "trained_model.pkl",
        output_dir: str = ".",
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.ml_filter = MLTradeFilter(
            model_path=model_path, csv_path=csv_path
        )
        self.position_sizer = PositionSizer()
        self.exit_optimizer = ExitOptimizer()

        # Results storage
        self.original_trades: Optional[pd.DataFrame] = None
        self.filtered_trades: Optional[pd.DataFrame] = None
        self.training_results: Dict = {}
        self.comparison: Dict = {}

    # ─────────────────────────────────────────────────────────────────────
    #  FULL PIPELINE
    # ─────────────────────────────────────────────────────────────────────

    def run_full_pipeline(self) -> Dict:
        """
        Execute the complete ML enhancement pipeline:
        1. Train ML model on historical CSV
        2. Apply ML filter to all trades
        3. Compare ML-filtered vs original
        4. Generate performance reports
        5. Save all outputs
        """
        log.info("=" * 70)
        log.info("  ML ENHANCEMENT PIPELINE — FULL RUN")
        log.info("=" * 70)

        # ── Step 1: Train ML model ────────────────────────────────────────
        log.info("\n  STEP 1: Training ML Trade Filter...")
        self.training_results = self.ml_filter.train(self.csv_path)

        # ── Step 2: Load & filter all trades ──────────────────────────────
        log.info("\n  STEP 2: Applying ML filter to all trades...")
        self.original_trades = self.ml_filter.load_data(self.csv_path)
        self.original_trades = engineer_features(self.original_trades)

        # Predict probabilities for ALL trades
        probs = self.ml_filter.predict_batch(self.original_trades)
        self.original_trades["ml_probability"] = probs
        self.original_trades["ml_confidence"] = self.original_trades[
            "ml_probability"
        ].apply(MLTradeFilter.classify_confidence)

        # ── Apply position sizing ─────────────────────────────────────────
        risk_multipliers = []
        for _, row in self.original_trades.iterrows():
            sizing = self.position_sizer.calculate_risk(
                balance=1000.0,
                probability=row["ml_probability"],
            )
            risk_multipliers.append(sizing["multiplier"])
        self.original_trades["risk_multiplier"] = risk_multipliers

        # ── Filter: keep only prob >= 0.40 ────────────────────────────────
        self.filtered_trades = self.original_trades[
            self.original_trades["ml_probability"] >= 0.40
        ].copy()

        log.info(
            f"  Original trades: {len(self.original_trades)} | "
            f"Filtered trades: {len(self.filtered_trades)} | "
            f"Rejected: {len(self.original_trades) - len(self.filtered_trades)}"
        )

        # ── Step 3: Compare performance ───────────────────────────────────
        log.info("\n  STEP 3: Performance comparison...")
        self.comparison = self._compare_performance()

        # ── Step 4: Generate reports ──────────────────────────────────────
        log.info("\n  STEP 4: Generating reports...")
        self._save_filtered_trades()
        self._save_performance_report()
        self._plot_comparison()
        self._plot_feature_importance()
        self._plot_confidence_vs_pnl()

        log.info("\n" + "=" * 70)
        log.info("  ML PIPELINE COMPLETE")
        log.info("=" * 70)

        return self.comparison

    # ─────────────────────────────────────────────────────────────────────
    #  PERFORMANCE COMPARISON
    # ─────────────────────────────────────────────────────────────────────

    def _compare_performance(self) -> Dict:
        """Compare original vs ML-filtered trade performance."""
        orig = self.original_trades
        filt = self.filtered_trades

        def calc_stats(df, label):
            n = len(df)
            if n == 0:
                return {"trades": 0, "win_rate": 0, "profit_factor": 0,
                        "total_pnl": 0, "avg_pnl": 0}

            wins = (df["outcome"] == 1).sum()
            losses = (df["outcome"] == 0).sum()
            wr = wins / n * 100

            # Compute PnL-based profit factor if pnl_usd exists
            if "pnl_usd" in df.columns:
                gross_profit = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
                gross_loss = abs(df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
                pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                total_pnl = df["pnl_usd"].sum()
                avg_pnl = df["pnl_usd"].mean()
            else:
                # Estimate from win/loss counts
                pf = (wins * 2.5) / losses if losses > 0 else float("inf")
                total_pnl = 0
                avg_pnl = 0

            # Drawdown from balance if available
            max_dd = 0.0
            if "balance" in df.columns and len(df) > 1:
                eq = pd.Series(df["balance"].values)
                dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
                max_dd = abs(dd)

            stats = {
                "label": label,
                "trades": n,
                "wins": int(wins),
                "losses": int(losses),
                "win_rate": round(wr, 2),
                "profit_factor": round(pf, 3),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 4),
                "max_drawdown": round(max_dd, 2),
            }
            return stats

        orig_stats = calc_stats(orig, "ORIGINAL (no ML)")
        filt_stats = calc_stats(filt, "ML-FILTERED (prob >= 0.40)")

        # High confidence subset
        high_conf = filt[filt["ml_probability"] > 0.55]
        high_stats = calc_stats(high_conf, "HIGH CONFIDENCE (prob > 0.55)")

        # Rejected trades analysis
        rejected = orig[orig["ml_probability"] < 0.40]
        rej_stats = calc_stats(rejected, "REJECTED (prob < 0.40)")

        comparison = {
            "original": orig_stats,
            "ml_filtered": filt_stats,
            "high_confidence": high_stats,
            "rejected": rej_stats,
        }

        # ── Print comparison table ────────────────────────────────────────
        log.info("\n" + "═" * 70)
        log.info(f"  {'METRIC':<25s} {'ORIGINAL':>12s} {'ML-FILTERED':>14s} {'HIGH CONF':>12s}")
        log.info("─" * 70)
        for metric in ["trades", "win_rate", "profit_factor", "total_pnl",
                       "max_drawdown"]:
            o = orig_stats.get(metric, "—")
            f = filt_stats.get(metric, "—")
            h = high_stats.get(metric, "—")
            log.info(f"  {metric:<25s} {str(o):>12s} {str(f):>14s} {str(h):>12s}")
        log.info("═" * 70)

        log.info(f"\n  Rejected trades: {len(rejected)}")
        log.info(f"  Rejected WR: {rej_stats['win_rate']}%")
        log.info(f"  → ML correctly avoided {rej_stats['losses']} losing trades")

        return comparison

    # ─────────────────────────────────────────────────────────────────────
    #  OUTPUT FILES
    # ─────────────────────────────────────────────────────────────────────

    def _save_filtered_trades(self):
        if self.filtered_trades is not None:
            path = os.path.join(self.output_dir, "filtered_trades.csv")
            export = self.filtered_trades.copy()
            # Keep key columns
            keep = [c for c in export.columns if c not in ["month", "year"]]
            export[keep].to_csv(path, index=False)
            log.info(f"  Saved filtered trades -> {path} ({len(export)} rows)")

    def _save_performance_report(self):
        report = {
            "training": {
                "total_trades": self.training_results.get("total_trades"),
                "train_size": self.training_results.get("train_size"),
                "test_size": self.training_results.get("test_size"),
                "train_metrics": self.training_results.get("train_metrics"),
                "test_metrics": self.training_results.get("test_metrics"),
                "oob_score": self.training_results.get("oob_score"),
            },
            "comparison": self.comparison,
            "feature_importances": self.training_results.get(
                "feature_importances", {}
            ),
            "walk_forward": self.training_results.get("walk_forward", []),
        }

        path = os.path.join(self.output_dir, "performance_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"  Saved performance report -> {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  CHARTS
    # ─────────────────────────────────────────────────────────────────────

    def _plot_comparison(self):
        """ML vs Non-ML equity curve comparison."""
        if self.original_trades is None or "balance" not in self.original_trades.columns:
            log.info("  No balance column — skipping equity comparison chart")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#0d1117")
        plt.rcParams.update({
            "text.color": "white", "axes.labelcolor": "white",
            "xtick.color": "grey", "ytick.color": "grey",
        })

        orig = self.original_trades
        filt = self.filtered_trades

        # ── 1. Equity curves ─────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(range(len(orig)), orig["balance"].values,
                color="#ff6b6b", alpha=0.7, label="Original", linewidth=1)
        if filt is not None and len(filt) > 0 and "balance" in filt.columns:
            # Recalculate filtered equity
            bal = 1000.0
            filt_equity = []
            for _, row in filt.iterrows():
                if "pnl_usd" in row:
                    mul = row.get("risk_multiplier", 1.0)
                    pnl = row["pnl_usd"] * mul
                    bal += pnl
                filt_equity.append(bal)
            ax.plot(range(len(filt_equity)), filt_equity,
                    color="#00c897", linewidth=1.5, label="ML-Filtered")
        ax.set_title("Equity Curve Comparison", fontsize=12, pad=8)
        ax.legend(fontsize=9)
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        # ── 2. Confidence distribution ────────────────────────────────────
        ax = axes[0, 1]
        ax.hist(orig["ml_probability"], bins=30, color="#4ecdc4",
                alpha=0.7, edgecolor="#333")
        ax.axvline(0.40, color="#ff6b6b", linestyle="--", label="Skip threshold")
        ax.axvline(0.55, color="#ffd93d", linestyle="--", label="High conf threshold")
        ax.set_title("ML Probability Distribution", fontsize=12, pad=8)
        ax.set_xlabel("Win Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        # ── 3. Win rate by confidence tier ────────────────────────────────
        ax = axes[1, 0]
        tiers = ["SKIP", "NORMAL", "HIGH_CONFIDENCE"]
        tier_wrs = []
        tier_counts = []
        for tier in tiers:
            subset = orig[orig["ml_confidence"] == tier]
            if len(subset) > 0:
                wr = (subset["outcome"] == 1).mean() * 100
            else:
                wr = 0
            tier_wrs.append(wr)
            tier_counts.append(len(subset))

        colors = ["#ff6b6b", "#ffd93d", "#00c897"]
        bars = ax.bar(tiers, tier_wrs, color=colors, edgecolor="#333")
        for bar, cnt in zip(bars, tier_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"n={cnt}",
                    ha="center", fontsize=9, color="white")
        ax.set_title("Win Rate by Confidence Tier", fontsize=12, pad=8)
        ax.set_ylabel("Win Rate (%)")
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        # ── 4. PnL distribution comparison ────────────────────────────────
        ax = axes[1, 1]
        if "pnl_usd" in orig.columns:
            ax.hist(orig["pnl_usd"], bins=40, color="#ff6b6b",
                    alpha=0.5, label="Original")
            if filt is not None and len(filt) > 0:
                ax.hist(filt["pnl_usd"], bins=40, color="#00c897",
                        alpha=0.5, label="ML-Filtered")
            ax.axvline(0, color="white", linewidth=0.5)
            ax.set_title("PnL Distribution", fontsize=12, pad=8)
            ax.set_xlabel("PnL ($)")
            ax.legend(fontsize=9)
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        fig.suptitle("ML Enhancement — Performance Comparison",
                     fontsize=14, y=0.999, color="white")
        plt.tight_layout()

        path = os.path.join(self.output_dir, "ml_comparison_chart.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Saved comparison chart -> {path}")

    def _plot_feature_importance(self):
        """Feature importance bar chart with SHAP-style visualization."""
        if self.ml_filter.feature_importances is None:
            return

        fi = self.ml_filter.feature_importances

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1117")
        plt.rcParams.update({
            "text.color": "white", "axes.labelcolor": "white",
            "xtick.color": "grey", "ytick.color": "grey",
        })

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi)))
        bars = ax.barh(range(len(fi)), fi.values[::-1],
                       color=colors, edgecolor="#333")
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi.index[::-1], fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title("Feature Importance (RandomForest)",
                     fontsize=13, pad=10)
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        # Add value labels
        for bar, val in zip(bars, fi.values[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, color="white")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Saved feature importance -> {path}")

    def _plot_confidence_vs_pnl(self):
        """Scatter: ML probability vs actual PnL."""
        if self.original_trades is None or "pnl_usd" not in self.original_trades.columns:
            return

        df = self.original_trades

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0d1117")
        plt.rcParams.update({
            "text.color": "white", "axes.labelcolor": "white",
            "xtick.color": "grey", "ytick.color": "grey",
        })

        colors = ["#00c897" if p > 0 else "#ff6b6b" for p in df["pnl_usd"]]
        ax.scatter(df["ml_probability"], df["pnl_usd"],
                   c=colors, alpha=0.5, s=15, edgecolors="none")
        ax.axvline(0.40, color="#ffd93d", linestyle="--",
                   alpha=0.7, label="Skip threshold (0.40)")
        ax.axvline(0.55, color="#4ecdc4", linestyle="--",
                   alpha=0.7, label="High conf (0.55)")
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("ML Win Probability", fontsize=11)
        ax.set_ylabel("Actual PnL ($)", fontsize=11)
        ax.set_title("Confidence vs PnL", fontsize=13, pad=10)
        ax.legend(fontsize=9)
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#333")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "confidence_vs_pnl.png")
        plt.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info(f"  Saved confidence vs PnL -> {path}")

    # ─────────────────────────────────────────────────────────────────────
    #  LIVE INTEGRATION HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def should_take_trade(self, features: dict) -> Dict:
        """
        Live trading integration point.
        Returns decision dict with probability, confidence, sizing.
        """
        prob = self.ml_filter.predict_trade_probability(features)
        confidence = MLTradeFilter.classify_confidence(prob)
        sizing = self.position_sizer.calculate_risk(
            balance=features.get("balance", 1000.0),
            probability=prob,
        )

        return {
            "probability": prob,
            "confidence": confidence,
            "should_trade": sizing["should_trade"],
            "risk_multiplier": sizing["multiplier"],
            "risk_usd": sizing["risk_usd"],
            "reason": sizing["reason"],
        }
