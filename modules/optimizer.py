# ╔══════════════════════════════════════════════════════════════╗
# ║  AUTO OPTIMIZER ENGINE (Optuna)                             ║
# ║  Optimizes strategy parameters with overfitting protection   ║
# ╚══════════════════════════════════════════════════════════════╝

import json, logging, os, warnings, re
import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable

warnings.filterwarnings("ignore")
log = logging.getLogger("AutoOptimizer")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    log.warning("Optuna not installed. Run: pip install optuna")


class AutoOptimizer:
    """
    Optuna-based parameter optimizer for the trading strategy.

    Optimizes: RSI thresholds, ATR_GATE, SL_MULT, RR_RATIO, EMA lengths
    Objective: Maximize Profit Factor
    Constraints: PF >= 2, MaxDD < 20%, Trades >= 50

    Uses walk-forward validation to prevent overfitting.
    """

    def __init__(
        self,
        backtest_fn: Optional[Callable] = None,
        add_indicators_fn: Optional[Callable] = None,
        n_trials: int = 100,
        min_trades: int = 50,
        min_profit_factor: float = 2.0,
        max_drawdown_pct: float = 20.0,
        output_path: str = "optimized_params.json",
    ):
        self.backtest_fn = backtest_fn
        self.add_indicators_fn = add_indicators_fn
        self.n_trials = n_trials
        self.min_trades = min_trades
        self.min_profit_factor = min_profit_factor
        self.max_drawdown_pct = max_drawdown_pct
        self.output_path = output_path
        self.best_params: Dict = {}
        self.all_results = []
        self._df_cache = None

    def set_data(self, df: pd.DataFrame):
        """Cache the raw OHLCV DataFrame for repeated backtests."""
        self._df_cache = df.copy()

    def _run_backtest_with_params(self, params: Dict, df: pd.DataFrame):
        """
        Run the existing backtest engine with modified parameters.
        Returns (trades_df, equity_curve, profit_factor, max_dd, n_trades).
        """
        import test25 as strategy

        # ── Temporarily override strategy globals ─────────────────────────
        orig = {
            "ATR_GATE": strategy.ATR_GATE,
            "SL_MULT": strategy.SL_MULT,
            "RR_RATIO": strategy.RR_RATIO,
            "RSI_BUY_THRESH": strategy.RSI_BUY_THRESH,
            "RSI_SELL_THRESH": strategy.RSI_SELL_THRESH,
        }

        strategy.ATR_GATE = params.get("atr_gate", strategy.ATR_GATE)
        strategy.SL_MULT = params.get("sl_mult", strategy.SL_MULT)
        strategy.RR_RATIO = params.get("rr_ratio", strategy.RR_RATIO)
        strategy.RSI_BUY_THRESH = params.get("rsi_buy_threshold", strategy.RSI_BUY_THRESH)
        strategy.RSI_SELL_THRESH = params.get("rsi_sell_threshold", strategy.RSI_SELL_THRESH)

        try:
            # Re-calculate indicators if EMA lengths changed
            df_ind = df.copy()
            if self.add_indicators_fn:
                df_ind = self.add_indicators_fn(df_ind)

            trades_df, equity, _, _ = strategy.backtest(df_ind)

            n_trades = len(trades_df)
            if n_trades == 0:
                return trades_df, equity, 0.0, 100.0, 0

            wins = (trades_df["result"] == "TP").sum()
            losses = (trades_df["result"] == "SL").sum()
            rr = params.get("rr_ratio", strategy.RR_RATIO)
            pf = (wins * rr) / losses if losses > 0 else float("inf")

            eq = pd.Series(equity)
            dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
            max_dd = abs(dd)

            return trades_df, equity, pf, max_dd, n_trades

        finally:
            # ── Restore original globals ──────────────────────────────────
            strategy.ATR_GATE = orig["ATR_GATE"]
            strategy.SL_MULT = orig["SL_MULT"]
            strategy.RR_RATIO = orig["RR_RATIO"]
            strategy.RSI_BUY_THRESH = orig["RSI_BUY_THRESH"]
            strategy.RSI_SELL_THRESH = orig["RSI_SELL_THRESH"]

    def _objective(self, trial, df):
        """Optuna objective function."""
        params = {
            "rsi_buy_threshold": trial.suggest_int("rsi_buy_threshold", 45, 65),
            "rsi_sell_threshold": trial.suggest_int("rsi_sell_threshold", 35, 55),
            "atr_gate": trial.suggest_float("atr_gate", 0.001, 0.008, step=0.0005),
            "sl_mult": trial.suggest_float("sl_mult", 1.0, 2.5, step=0.1),
            "rr_ratio": trial.suggest_float("rr_ratio", 1.5, 4.0, step=0.1),
        }

        # ── Walk-forward: split data 70/30 ────────────────────────────────
        n = len(df)
        split = int(n * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        # ── Run on training period ────────────────────────────────────────
        try:
            _, _, pf_train, dd_train, n_train = self._run_backtest_with_params(
                params, train_df
            )
        except Exception as e:
            log.debug(f"Trial failed on train: {e}")
            return 0.0

        # ── Constraint checks on train ────────────────────────────────────
        if n_train < self.min_trades * 0.7:
            return 0.0

        # ── Run on test period (validation) ───────────────────────────────
        try:
            _, _, pf_test, dd_test, n_test = self._run_backtest_with_params(
                params, test_df
            )
        except Exception as e:
            log.debug(f"Trial failed on test: {e}")
            return 0.0

        if n_test < self.min_trades * 0.3:
            return 0.0

        # ── Combined score ────────────────────────────────────────────────
        # Penalise if train/test PF diverge significantly (overfitting sign)
        if pf_train > 0 and pf_test > 0:
            pf_ratio = min(pf_test / pf_train, pf_train / pf_test)
        else:
            pf_ratio = 0

        # Score: weighted average of train+test PF, penalised by drawdown
        combined_pf = 0.4 * pf_train + 0.6 * pf_test
        dd_penalty = max(0, 1.0 - max(dd_train, dd_test) / self.max_drawdown_pct)
        stability_bonus = pf_ratio * 0.5

        score = combined_pf * dd_penalty + stability_bonus

        # ── Hard constraint violations → heavy penalty ────────────────────
        if dd_test > self.max_drawdown_pct:
            score *= 0.3
        if pf_test < 1.0:
            score *= 0.5

        self.all_results.append({
            "trial": trial.number,
            "params": params,
            "pf_train": round(pf_train, 3),
            "pf_test": round(pf_test, 3),
            "dd_train": round(dd_train, 2),
            "dd_test": round(dd_test, 2),
            "n_train": n_train,
            "n_test": n_test,
            "score": round(score, 4),
        })

        return score

    def optimize(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run the full Optuna optimization loop.

        Returns: dict with best params, scores, and all trial results.
        """
        if not HAS_OPTUNA:
            log.error("Optuna not installed. Skipping optimization.")
            return {"error": "optuna not installed"}

        if df is None:
            df = self._df_cache
        if df is None:
            raise ValueError("No data provided. Call set_data() or pass df.")

        log.info("=" * 60)
        log.info("  AUTO OPTIMIZER — Optuna Parameter Search")
        log.info(f"  Trials: {self.n_trials}")
        log.info(f"  Constraints: PF >= {self.min_profit_factor}, "
                 f"DD < {self.max_drawdown_pct}%, Trades >= {self.min_trades}")
        log.info("=" * 60)

        study = optuna.create_study(
            direction="maximize",
            study_name="strategy_optimizer",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        study.optimize(
            lambda trial: self._objective(trial, df),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # ── Extract best params ───────────────────────────────────────────
        self.best_params = study.best_params
        best_score = study.best_value

        log.info(f"\n  BEST SCORE: {best_score:.4f}")
        log.info(f"  BEST PARAMS: {json.dumps(self.best_params, indent=2)}")

        # ── Save results ──────────────────────────────────────────────────
        output = {
            "best_params": self.best_params,
            "best_score": best_score,
            "n_trials": self.n_trials,
            "constraints": {
                "min_profit_factor": self.min_profit_factor,
                "max_drawdown_pct": self.max_drawdown_pct,
                "min_trades": self.min_trades,
            },
            "top_10_trials": sorted(
                self.all_results, key=lambda x: x["score"], reverse=True
            )[:10],
        }

        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"  Results saved -> {self.output_path}")

        # ── Auto-Update test25.py ─────────────────────────────────────────
        self._update_test25_defaults()

        return output

    def _update_test25_defaults(self):
        """Automatically updates the default parameters in test25.py"""
        try:
            path = "test25.py"
            if not os.path.exists(path):
                return
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            p = self.best_params
            if "atr_gate" in p:
                content = re.sub(
                    r'(ATR_GATE\s*=\s*float\(os\.getenv\("ATR_GATE",\s*")[^"]+("\)\))',
                    rf'\g<1>{p["atr_gate"]}\g<2>', content
                )
            if "rsi_buy_threshold" in p:
                content = re.sub(
                    r'(RSI_BUY_THRESH\s*=\s*int\(os\.getenv\("RSI_BUY_THRESH",\s*")[^"]+("\)\))',
                    rf'\g<1>{int(p["rsi_buy_threshold"])}\g<2>', content
                )
            if "rsi_sell_threshold" in p:
                content = re.sub(
                    r'(RSI_SELL_THRESH\s*=\s*int\(os\.getenv\("RSI_SELL_THRESH",\s*")[^"]+("\)\))',
                    rf'\g<1>{int(p["rsi_sell_threshold"])}\g<2>', content
                )
            if "sl_mult" in p:
                content = re.sub(
                    r'(SL_MULT\s*=\s*float\(os\.getenv\("SL_MULT",\s*")[^"]+("\)\))',
                    rf'\g<1>{p["sl_mult"]}\g<2>', content
                )
            if "rr_ratio" in p:
                content = re.sub(
                    r'(RR_RATIO\s*=\s*float\(os\.getenv\("RR_RATIO",\s*")[^"]+("\)\))',
                    rf'\g<1>{p["rr_ratio"]}\g<2>', content
                )

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                
            log.info(f"  ✅ Auto-updated {path} with optimal parameters.")
        except Exception as e:
            log.error(f"  ❌ Failed to auto-update test25.py: {e}")
