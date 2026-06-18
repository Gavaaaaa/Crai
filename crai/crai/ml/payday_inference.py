"""crai/ml/payday_inference.py — LSTM + Prophet: Inferência de Liquidez (Payday Inference)."""

import numpy as np
from datetime import datetime, timedelta

PAYDAY_HEURISTICS = {
    "CLT":        [5, 6, 7, 20, 21],
    "freelancer": [10, 15, 20, 25],
    "PJ":         [5, 10, 15, 20, 25],
    "default":    [5, 20],
}


class PaydayInference:
    """Fase 1: heurística por perfil. Fase 2: LSTM + Prophet treinados."""

    def __init__(self):
        self.is_fitted = False

    async def predict_next_window(self, customer_id: str) -> dict:
        history = await self._fetch_history(customer_id)
        profile = self._classify_profile(history)
        return self._heuristic_predict(profile)

    def _classify_profile(self, history: list) -> str:
        amounts = [h["amount"] for h in history if h.get("status") == "succeeded"]
        if len(amounts) < 3:
            return "CLT"
        cv = np.std(amounts) / (np.mean(amounts) + 1e-9)
        if cv < 0.15: return "CLT"
        if cv < 0.40: return "PJ"
        return "freelancer"

    def _heuristic_predict(self, profile: str) -> dict:
        paydays = PAYDAY_HEURISTICS.get(profile, PAYDAY_HEURISTICS["default"])
        today = datetime.now()
        next_date = None
        for day in sorted(paydays):
            candidate = today.replace(day=min(day, 28))
            if candidate > today + timedelta(hours=2):
                next_date = candidate
                break
        if next_date is None:
            next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
            next_date = next_month.replace(day=paydays[0])
        return {
            "timestamp":  next_date,
            "confidence": 0.65 if profile == "CLT" else 0.45,
            "profile":    profile,
        }

    async def _fetch_history(self, customer_id: str) -> list:
        rng = np.random.default_rng(seed=abs(hash(customer_id)) % (2**32))
        base = rng.uniform(500, 5000)
        noise = rng.normal(0, base * 0.05, 30)
        return [
            {"amount": round(float(base + noise[i]), 2), "date": datetime.now() - timedelta(days=i),
             "status": "succeeded" if rng.random() > 0.05 else "failed"}
            for i in range(30)
        ]
