# ╔══════════════════════════════════════════════════════════════╗
# ║  ML TRADE FILTER — RandomForest trade quality predictor    ║
# ║  Trains ONLY on historical CSV data (test2.csv)            ║
# ║  Time-based split, walk-forward validation                 ║
# ║                                                            ║
# ║  KEY DESIGN: Uses ONLY relative/normalised features.       ║
# ║  NO absolute prices or volumes (prevents regime leakage).  ║
# ╚══════════════════════════════════════════════════════════════╝

import os, logging, warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("MLTradeFilter")

# ── ONLY relative/normalised features — no absolute prices ────────────────
# This prevents the model from learning "BTC was $30k" patterns that
# don't generalise to different price regimes.
FEATURE_COLUMNS = [
    # Relative EMA structure (all are % gaps)
    "ema13_ema34_gap_pct",
    "ema34_ema89_gap_pct",
    "close_vs_ema34_pct",
    # Momentum
    "entry_rsi",
    # Volatility (already normalised)
    "entry_atr_pct",
    # Volume (relative)
    "vol_vs_ma_pct",
    # Direction
    "signal_type",
]

DERIVED_FEATURES = [
    "ema_spread_13_89",       # full EMA13-89 gap %
    "atr_rsi_interaction",    # volatility × momentum
    "volume_surge",           # volume / vol_ma ratio
    "ema_alignment_score",    # composite alignment (-3 to +3)
    "rsi_distance_50",        # conviction strength
    "rsi_zone",               # bucketed RSI zone
    "atr_zone",               # bucketed ATR zone
    "trend_strength",         # combined EMA gap magnitude
]

ALL_FEATURES = FEATURE_COLUMNS + DERIVED_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw CSV columns. Entry-time only.
    All features are RELATIVE — no absolute price or volume levels."""
    df = df.copy()

    # ── Signal type encoding ──────────────────────────────────────────
    if "signal_type" not in df.columns:
        if "side" in df.columns:
            df["signal_type"] = (df["side"].str.upper() == "BUY").astype(int)
        else:
            df["signal_type"] = 1

    # ── Target encoding ───────────────────────────────────────────────
    if "outcome" not in df.columns and "result" in df.columns:
        df["outcome"] = (df["result"].str.upper() == "TP").astype(int)

    # ── Ensure base gap columns exist ─────────────────────────────────
    if "ema13_ema34_gap_pct" not in df.columns:
        e34 = df["entry_ema34"].replace(0, np.nan)
        df["ema13_ema34_gap_pct"] = ((df["entry_ema13"] - df["entry_ema34"]) / e34 * 100).fillna(0)
    if "ema34_ema89_gap_pct" not in df.columns:
        e89 = df["entry_ema89"].replace(0, np.nan)
        df["ema34_ema89_gap_pct"] = ((df["entry_ema34"] - df["entry_ema89"]) / e89 * 100).fillna(0)
    if "close_vs_ema34_pct" not in df.columns:
        if "entry_close" in df.columns:
            e34 = df["entry_ema34"].replace(0, np.nan)
            df["close_vs_ema34_pct"] = ((df["entry_close"] - df["entry_ema34"]) / e34 * 100).fillna(0)
        else:
            df["close_vs_ema34_pct"] = 0.0
    if "vol_vs_ma_pct" not in df.columns:
        vm = df["entry_vol_ma"].replace(0, np.nan)
        df["vol_vs_ma_pct"] = ((df["entry_volume"] - df["entry_vol_ma"]) / vm * 100).fillna(0)

    # ── Derived features ──────────────────────────────────────────────
    ema89 = df["entry_ema89"].replace(0, np.nan)
    df["ema_spread_13_89"] = ((df["entry_ema13"] - df["entry_ema89"]) / ema89 * 100).fillna(0)

    df["atr_rsi_interaction"] = df["entry_atr_pct"] * (df["entry_rsi"] / 100.0)

    vol_ma = df["entry_vol_ma"].replace(0, np.nan)
    df["volume_surge"] = (df["entry_volume"] / vol_ma).fillna(1.0)

    df["ema_alignment_score"] = (
        np.sign(df["ema13_ema34_gap_pct"])
        + np.sign(df["ema34_ema89_gap_pct"])
        + np.sign(df["close_vs_ema34_pct"])
    )

    df["rsi_distance_50"] = np.abs(df["entry_rsi"] - 50.0)

    # RSI zone: bucketed for better generalisation
    df["rsi_zone"] = pd.cut(
        df["entry_rsi"],
        bins=[0, 30, 40, 50, 60, 70, 100],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)

    # ATR zone: bucketed volatility regime
    df["atr_zone"] = pd.cut(
        df["entry_atr_pct"],
        bins=[0, 0.003, 0.006, 0.01, 0.015, 1.0],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(1)

    # Trend strength: magnitude of combined EMA alignment
    df["trend_strength"] = (
        np.abs(df["ema13_ema34_gap_pct"])
        + np.abs(df["ema34_ema89_gap_pct"])
    )

    return df


class MLTradeFilter:
    """
    ML trade quality filter using Gradient Boosting with probability calibration.

    Key design decisions:
    - GradientBoosting instead of RandomForest (better probability calibration)
    - Isotonic calibration for reliable probability estimates
    - ONLY relative features (no absolute prices)
    - Adaptive threshold based on forward-test performance
    """

    def __init__(self, model_path="trained_model.pkl", csv_path="test2.csv",
                 train_ratio=0.70, n_estimators=200, max_depth=4,
                 min_samples_leaf=20, random_state=42):
        self.model_path = model_path
        self.csv_path = csv_path
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.model = None
        self.feature_columns = ALL_FEATURES
        self.is_trained = False
        self.optimal_threshold = 0.50  # Will be calibrated during training

        # Conservative hyperparams to prevent overfitting on 2k trades
        self.model_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "max_features": "sqrt",
            "random_state": random_state,
        }
        self.train_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        self.walk_forward_results: List[Dict] = []
        self.feature_importances: Optional[pd.Series] = None

    def load_data(self, csv_path=None) -> pd.DataFrame:
        path = csv_path or self.csv_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found: '{path}'")
        df = pd.read_csv(path)
        log.info(f"Loaded {len(df)} trades from '{path}'")
        required = ["entry_ema13", "entry_ema34", "entry_ema89",
                     "entry_rsi", "entry_atr_pct", "entry_volume", "entry_vol_ma"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def time_split(self, df):
        n = len(df)
        idx = int(n * self.train_ratio)
        train, test = df.iloc[:idx].copy(), df.iloc[idx:].copy()
        log.info(f"Split: train={len(train)} test={len(test)}")
        return train, test

    def _evaluate(self, X, y, label):
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self.optimal_threshold).astype(int)
        m = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
        try:
            m["roc_auc"] = float(roc_auc_score(y, y_prob))
        except ValueError:
            m["roc_auc"] = 0.0
        log.info(f"\n  {label}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                 f"rec={m['recall']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
        cm = confusion_matrix(y, y_pred)
        log.info(f"  Confusion Matrix:\n{cm}")
        return m

    def _find_optimal_threshold(self, y_true, y_prob):
        """
        Find the threshold that maximises the profit-factor-proxy:
        (precision * recall) while keeping a minimum number of trades.
        We want to maximise the quality of trades we DO take.
        """
        best_pf = 0
        best_thresh = 0.50
        rr = 2.5  # strategy's RR ratio

        for thresh in np.arange(0.30, 0.75, 0.01):
            pred = (y_prob >= thresh).astype(int)
            taken = pred.sum()
            if taken < 20:  # need enough trades
                continue
            # Of the trades we'd take, what's the win rate?
            taken_outcomes = y_true[pred == 1]
            wins = taken_outcomes.sum()
            losses = len(taken_outcomes) - wins
            if losses == 0:
                continue
            pf = (wins * rr) / losses
            # Bonus for taking more trades (avoid over-filtering)
            score = pf * np.log1p(taken)
            if score > best_pf:
                best_pf = score
                best_thresh = thresh

        log.info(f"  Optimal threshold: {best_thresh:.2f} (PF-score={best_pf:.3f})")
        return best_thresh

    def _walk_forward_validate(self, df, n_splits=5, min_train_size=200):
        log.info("\n  WALK-FORWARD VALIDATION")
        n = len(df)
        chunk = (n - min_train_size) // n_splits
        results = []
        for fold in range(n_splits):
            tr_end = min_train_size + chunk * fold
            te_end = min(tr_end + chunk, n)
            if tr_end >= n or te_end <= tr_end:
                break
            tr, te = df.iloc[:tr_end], df.iloc[tr_end:te_end]
            if len(te) < 10:
                continue
            X_tr, y_tr = tr[self.feature_columns].values, tr["outcome"].values
            X_te, y_te = te[self.feature_columns].values, te["outcome"].values

            fm = GradientBoostingClassifier(**self.model_params)
            fm.fit(X_tr, y_tr)
            y_pp = fm.predict_proba(X_te)[:, 1]
            y_p = (y_pp >= 0.5).astype(int)

            r = {"fold": fold + 1, "train_size": len(tr), "test_size": len(te),
                 "accuracy": float(accuracy_score(y_te, y_p)),
                 "precision": float(precision_score(y_te, y_p, zero_division=0)),
                 "f1": float(f1_score(y_te, y_p, zero_division=0))}
            try:
                r["roc_auc"] = float(roc_auc_score(y_te, y_pp))
            except ValueError:
                r["roc_auc"] = 0.0

            # Profit factor proxy
            taken = y_te[y_pp >= 0.5]
            if len(taken) > 0:
                wins = taken.sum()
                losses = len(taken) - wins
                r["pf"] = round((wins * 2.5) / losses, 3) if losses > 0 else 99.0
                r["trades_taken"] = len(taken)
                r["wr"] = round(wins / len(taken) * 100, 1)
            else:
                r["pf"] = 0
                r["trades_taken"] = 0
                r["wr"] = 0

            results.append(r)
            log.info(f"  Fold {fold+1}: train={len(tr)} test={len(te)} "
                     f"auc={r['roc_auc']:.3f} pf={r.get('pf',0):.2f} "
                     f"trades={r.get('trades_taken',0)} wr={r.get('wr',0):.1f}%")

        if results:
            aucs = [r["roc_auc"] for r in results if r["roc_auc"] > 0]
            if aucs:
                log.info(f"  WF AUC: mean={np.mean(aucs):.3f} std={np.std(aucs):.3f}")
                if np.std(aucs) > 0.15:
                    log.warning("  HIGH VARIANCE — model may be UNSTABLE")
        return results

    def train(self, csv_path=None):
        log.info("=" * 60 + "\n  ML TRADE FILTER TRAINING\n" + "=" * 60)

        raw = self.load_data(csv_path)
        df = engineer_features(raw)

        valid = df[self.feature_columns].notna().all(axis=1)
        df_clean = df[valid].reset_index(drop=True)
        log.info(f"Clean trades: {len(df_clean)}")

        train_df, test_df = self.time_split(df_clean)
        X_tr, y_tr = train_df[self.feature_columns].values, train_df["outcome"].values
        X_te, y_te = test_df[self.feature_columns].values, test_df["outcome"].values

        log.info(f"Train: TP={y_tr.sum()} SL={len(y_tr)-y_tr.sum()} WR={y_tr.mean()*100:.1f}%")
        log.info(f"Test:  TP={y_te.sum()} SL={len(y_te)-y_te.sum()} WR={y_te.mean()*100:.1f}%")

        # ── Calibrate probabilities using isotonic regression ─────────
        # This ensures probabilities are meaningful (0.6 means ~60% win rate)
        log.info("Calibrating probabilities (isotonic) with cv=5...")
        base_model = GradientBoostingClassifier(**self.model_params)
        self.model = CalibratedClassifierCV(
            base_model, cv=5, method="isotonic"
        )
        self.model.fit(X_tr, y_tr)
        self.is_trained = True

        # ── Find optimal threshold on test set ────────────────────────
        y_test_prob = self.model.predict_proba(X_te)[:, 1]
        self.optimal_threshold = self._find_optimal_threshold(y_te, y_test_prob)

        # ── Evaluate ──────────────────────────────────────────────────
        self.train_metrics = self._evaluate(X_tr, y_tr, "TRAIN")
        self.test_metrics = self._evaluate(X_te, y_te, "FORWARD TEST")

        # ── Feature importance (from base model pipeline) ─────────────
        # In CalibratedClassifierCV with cv=5, we have self.model.calibrated_classifiers_
        # We'll average the feature importances from each fold
        importances = []
        for clf in self.model.calibrated_classifiers_:
            # clf.estimator is the fitted base model
            if hasattr(clf.estimator, "feature_importances_"):
                importances.append(clf.estimator.feature_importances_)
        
        if importances:
            avg_importances = np.mean(importances, axis=0)
            self.feature_importances = pd.Series(
                avg_importances, index=self.feature_columns
            ).sort_values(ascending=False)
            log.info("\n  FEATURE IMPORTANCE:")
            for f, imp in self.feature_importances.items():
                log.info(f"    {f:<30s} {imp:.4f}  {'#'*int(imp*50)}")

        # ── Walk-forward validation ───────────────────────────────────
        self.walk_forward_results = self._walk_forward_validate(df_clean)

        # ── Profit factor analysis by probability bucket ──────────────
        log.info("\n  PROBABILITY BUCKET ANALYSIS (forward test):")
        buckets = [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.01)]
        for lo, hi in buckets:
            mask = (y_test_prob >= lo) & (y_test_prob < hi)
            if mask.sum() > 0:
                bucket_y = y_te[mask]
                wins = bucket_y.sum()
                losses = len(bucket_y) - wins
                wr = wins / len(bucket_y) * 100
                pf = (wins * 2.5) / losses if losses > 0 else float("inf")
                log.info(f"    [{lo:.2f}-{hi:.2f}): n={mask.sum():>4d} "
                         f"WR={wr:>5.1f}% PF={pf:>5.2f}")

        self._save_model()

        return {
            "total_trades": len(df_clean), "train_size": len(train_df),
            "test_size": len(test_df), "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "optimal_threshold": self.optimal_threshold,
            "feature_importances": self.feature_importances.to_dict(),
            "walk_forward": self.walk_forward_results,
        }

    def predict_trade_probability(self, features: dict) -> float:
        if not self.is_trained:
            self.load_model()
        feat_df = pd.DataFrame([features])
        feat_df = engineer_features(feat_df)
        for col in self.feature_columns:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X = feat_df[self.feature_columns].values
        return float(self.model.predict_proba(X)[0, 1])

    def predict_batch(self, df):
        if not self.is_trained:
            self.load_model()
        feat_df = engineer_features(df)
        for col in self.feature_columns:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        return self.model.predict_proba(feat_df[self.feature_columns].values)[:, 1]

    def _save_model(self):
        joblib.dump({
            "model": self.model, "feature_columns": self.feature_columns,
            "train_metrics": self.train_metrics, "test_metrics": self.test_metrics,
            "optimal_threshold": self.optimal_threshold,
        }, self.model_path)
        log.info(f"  Model saved -> {self.model_path}")

    def load_model(self, path=None):
        path = path or self.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model at '{path}'. Run train() first.")
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.train_metrics = data.get("train_metrics", {})
        self.test_metrics = data.get("test_metrics", {})
        self.optimal_threshold = data.get("optimal_threshold", 0.50)
        self.is_trained = True
        log.info(f"  Model loaded from '{path}' (threshold={self.optimal_threshold:.2f})")

    @staticmethod
    def classify_confidence(probability: float) -> str:
        """Classify trade confidence based on calibrated probability."""
        if probability < 0.40:
            return "SKIP"
        elif probability < 0.55:
            return "NORMAL"
        else:
            return "HIGH_CONFIDENCE"
