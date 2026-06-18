"""crai/ml/anomaly_detector.py — Autoencoder: detecção de anomalias não supervisionada."""

import numpy as np


class AnomalyDetector:
    """Cold start: heurística de z-score. Produção: autoencoder por cliente."""

    THRESHOLDS = {"CLT": 0.15, "PJ": 0.35, "freelancer": 0.55, "default": 0.30}

    def __init__(self):
        self.is_fitted = False

    async def check(self, customer_id: str, event: dict) -> dict:
        features  = self._extract_features(event)
        profile   = event.get("profile_type", "default")
        threshold = self.THRESHOLDS.get(profile, self.THRESHOLDS["default"])
        error     = self._simple_outlier_score(features)
        return {
            "is_anomaly": error > threshold,
            "error":      round(error, 4),
            "threshold":  threshold,
        }

    def _simple_outlier_score(self, features: np.ndarray) -> float:
        if len(features) == 0:
            return 0.1
        mean = np.mean(features)
        std  = np.std(features) + 1e-9
        z    = np.abs((features - mean) / std)
        return float(np.clip(np.mean(z) / 5.0, 0, 1))

    def _extract_features(self, event: dict) -> np.ndarray:
        charge = event.get("data", {}).get("object", {})
        return np.array([
            charge.get("amount", 0) / 100,
            len(charge.get("failure_code", "") or ""),
            int(bool(charge.get("failure_message"))),
            charge.get("attempt_count", 1),
        ], dtype=float)
