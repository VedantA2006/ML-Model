# ╔══════════════════════════════════════════════════════════════╗
# ║  POSITION SIZING ENGINE                                     ║
# ║  Dynamic risk allocation based on ML confidence              ║
# ╚══════════════════════════════════════════════════════════════╝

import logging
from typing import Dict

log = logging.getLogger("PositionSizer")


class PositionSizer:
    """
    Adjusts position risk based on ML confidence probability.

    Confidence tiers:
      > 0.85  → HIGH_CONFIDENCE → 1.5x risk multiplier
      0.70–0.85 → NORMAL → 1.0x risk multiplier
      < 0.70 → SKIP → 0x (trade rejected)
    """

    def __init__(
        self,
        base_risk_pct: float = 0.01,
        high_confidence_mult: float = 1.5,
        normal_mult: float = 1.0,
        skip_threshold: float = 0.40,
        high_threshold: float = 0.55,
        max_risk_pct: float = 0.03,
        max_concurrent_risk_pct: float = 0.06,
    ):
        self.base_risk_pct = base_risk_pct
        self.high_confidence_mult = high_confidence_mult
        self.normal_mult = normal_mult
        self.skip_threshold = skip_threshold
        self.high_threshold = high_threshold
        self.max_risk_pct = max_risk_pct
        self.max_concurrent_risk_pct = max_concurrent_risk_pct

    def calculate_risk(
        self,
        balance: float,
        probability: float,
        current_open_risk: float = 0.0,
    ) -> Dict:
        """
        Calculate position risk amount based on ML probability.

        Args:
            balance: current account balance
            probability: ML predicted win probability (0-1)
            current_open_risk: total risk already allocated to open trades

        Returns:
            dict with risk_usd, risk_pct, multiplier, confidence, should_trade
        """
        # ── Determine confidence tier ─────────────────────────────────────
        if probability < self.skip_threshold:
            return {
                "should_trade": False,
                "confidence": "SKIP",
                "probability": probability,
                "multiplier": 0.0,
                "risk_pct": 0.0,
                "risk_usd": 0.0,
                "reason": f"Probability {probability:.3f} < {self.skip_threshold} threshold",
            }

        if probability > self.high_threshold:
            confidence = "HIGH_CONFIDENCE"
            multiplier = self.high_confidence_mult
        else:
            confidence = "NORMAL"
            multiplier = self.normal_mult

        # ── Calculate risk ────────────────────────────────────────────────
        raw_risk_pct = self.base_risk_pct * multiplier
        capped_risk_pct = min(raw_risk_pct, self.max_risk_pct)

        # ── Check concurrent risk limit ───────────────────────────────────
        total_risk_pct = (current_open_risk / balance) + capped_risk_pct if balance > 0 else 0
        if total_risk_pct > self.max_concurrent_risk_pct:
            remaining = max(0, self.max_concurrent_risk_pct - (current_open_risk / balance if balance > 0 else 0))
            capped_risk_pct = min(capped_risk_pct, remaining)
            if capped_risk_pct <= 0:
                return {
                    "should_trade": False,
                    "confidence": confidence,
                    "probability": probability,
                    "multiplier": multiplier,
                    "risk_pct": 0.0,
                    "risk_usd": 0.0,
                    "reason": "Concurrent risk limit exceeded",
                }

        risk_usd = balance * capped_risk_pct

        result = {
            "should_trade": True,
            "confidence": confidence,
            "probability": probability,
            "multiplier": multiplier,
            "risk_pct": round(capped_risk_pct, 6),
            "risk_usd": round(risk_usd, 4),
            "reason": f"{confidence} trade at {probability:.3f} prob",
        }

        log.info(
            f"  Position: {confidence} prob={probability:.3f} "
            f"mult={multiplier}x risk=${risk_usd:.2f} ({capped_risk_pct*100:.2f}%)"
        )
        return result

    def calculate_position_size(
        self,
        balance: float,
        probability: float,
        entry_price: float,
        sl_price: float,
        current_open_risk: float = 0.0,
    ) -> Dict:
        """
        Calculate exact position size in units given entry and SL prices.
        """
        risk_info = self.calculate_risk(balance, probability, current_open_risk)
        if not risk_info["should_trade"]:
            risk_info["position_size"] = 0.0
            return risk_info

        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0:
            risk_info["position_size"] = 0.0
            risk_info["should_trade"] = False
            risk_info["reason"] = "Invalid SL distance (zero or negative)"
            return risk_info

        position_size = risk_info["risk_usd"] / sl_distance
        risk_info["position_size"] = round(position_size, 8)
        risk_info["sl_distance"] = round(sl_distance, 4)
        return risk_info
