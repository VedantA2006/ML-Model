# ╔══════════════════════════════════════════════════════════════╗
# ║  EXIT OPTIMIZATION ENGINE                                   ║
# ║  Dynamic exit decisions via rule-based + ML hybrid           ║
# ╚══════════════════════════════════════════════════════════════╝

import logging
import numpy as np
from typing import Dict, Optional

log = logging.getLogger("ExitOptimizer")


class ExitOptimizer:
    """
    Hybrid exit optimizer combining rule-based heuristics with
    adaptive logic to decide HOLD vs EXIT for open positions.

    Inputs at each check:
      - time_in_trade (hours)
      - current_pnl_pct (unrealised P&L %)
      - current_rsi
      - current_atr_pct
      - entry_atr_pct (at trade open)
      - side (BUY / SELL)

    Output: {"action": "HOLD" | "EXIT", "reason": str}
    """

    def __init__(
        self,
        max_hold_hours: float = 72.0,
        trailing_activate_pct: float = 1.5,
        trailing_stop_pct: float = 0.5,
        rsi_exit_buy_threshold: float = 78.0,
        rsi_exit_sell_threshold: float = 22.0,
        atr_spike_mult: float = 2.0,
        breakeven_hours: float = 24.0,
        min_profit_after_hours: float = 0.3,
        stagnation_hours: float = 48.0,
        stagnation_pnl_threshold: float = 0.2,
    ):
        self.max_hold_hours = max_hold_hours
        self.trailing_activate_pct = trailing_activate_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.rsi_exit_buy = rsi_exit_buy_threshold
        self.rsi_exit_sell = rsi_exit_sell_threshold
        self.atr_spike_mult = atr_spike_mult
        self.breakeven_hours = breakeven_hours
        self.min_profit_after_hours = min_profit_after_hours
        self.stagnation_hours = stagnation_hours
        self.stagnation_pnl_threshold = stagnation_pnl_threshold

        # Track peak PnL for trailing logic
        self._peak_pnl: Dict[str, float] = {}

    def evaluate_exit(
        self,
        trade_id: str,
        side: str,
        time_in_trade: float,
        current_pnl_pct: float,
        current_rsi: float,
        current_atr_pct: float,
        entry_atr_pct: float,
    ) -> Dict:
        """
        Evaluate whether to exit a trade early.

        Returns: {"action": "HOLD"|"EXIT", "reason": str, "priority": int}
        """
        # ── Rule 1: Maximum hold time exceeded ────────────────────────────
        if time_in_trade >= self.max_hold_hours:
            self._cleanup(trade_id)
            return {"action": "EXIT", "reason": f"Max hold time ({self.max_hold_hours}h) exceeded",
                    "priority": 1}

        # ── Rule 2: RSI extreme reversal signal ──────────────────────────
        if side == "BUY" and current_rsi >= self.rsi_exit_buy:
            if current_pnl_pct > 0:
                self._cleanup(trade_id)
                return {"action": "EXIT", "reason": f"RSI={current_rsi:.1f} overbought + profitable",
                        "priority": 2}

        if side == "SELL" and current_rsi <= self.rsi_exit_sell:
            if current_pnl_pct > 0:
                self._cleanup(trade_id)
                return {"action": "EXIT", "reason": f"RSI={current_rsi:.1f} oversold + profitable",
                        "priority": 2}

        # ── Rule 3: ATR spike — volatility regime change ──────────────────
        if entry_atr_pct > 0 and current_atr_pct > entry_atr_pct * self.atr_spike_mult:
            if current_pnl_pct > 0:
                self._cleanup(trade_id)
                return {"action": "EXIT",
                        "reason": f"ATR spike {current_atr_pct:.4f} > {entry_atr_pct*self.atr_spike_mult:.4f}",
                        "priority": 3}

        # ── Rule 4: Trailing stop logic ───────────────────────────────────
        peak = self._peak_pnl.get(trade_id, current_pnl_pct)
        if current_pnl_pct > peak:
            peak = current_pnl_pct
            self._peak_pnl[trade_id] = peak

        if peak >= self.trailing_activate_pct:
            drawback = peak - current_pnl_pct
            if drawback >= self.trailing_stop_pct:
                self._cleanup(trade_id)
                return {"action": "EXIT",
                        "reason": f"Trailing stop: peak={peak:.2f}% current={current_pnl_pct:.2f}%",
                        "priority": 4}

        # ── Rule 5: Stagnation check ─────────────────────────────────────
        if (time_in_trade >= self.stagnation_hours
                and abs(current_pnl_pct) < self.stagnation_pnl_threshold):
            self._cleanup(trade_id)
            return {"action": "EXIT",
                    "reason": f"Stagnation: {time_in_trade:.0f}h with only {current_pnl_pct:.2f}% PnL",
                    "priority": 5}

        # ── Rule 6: Breakeven timeout ─────────────────────────────────────
        if (time_in_trade >= self.breakeven_hours
                and current_pnl_pct < self.min_profit_after_hours):
            self._cleanup(trade_id)
            return {"action": "EXIT",
                    "reason": f"Breakeven timeout: {time_in_trade:.0f}h, PnL={current_pnl_pct:.2f}%",
                    "priority": 6}

        # ── Default: HOLD ─────────────────────────────────────────────────
        return {"action": "HOLD", "reason": "No exit condition met", "priority": 99}

    def _cleanup(self, trade_id: str):
        self._peak_pnl.pop(trade_id, None)

    def reset(self):
        self._peak_pnl.clear()
