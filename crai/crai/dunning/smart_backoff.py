"""crai/dunning/smart_backoff.py — Backoff Exponencial com Jitter."""

import random
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class BackoffConfig:
    base_hours: float
    max_attempts: int
    jitter_minutes: int


BACKOFF_RULES = {
    "insufficient_funds": BackoffConfig(1.0, 4, 10),
    "technical_error":    BackoffConfig(0.083, 3, 5),
    "security_block":     BackoffConfig(2.0, 2, 20),
    "card_expired":       BackoffConfig(0.0, 1, 0),
}


class SmartBackoff:
    def get_schedule(self, cause: str, attempt: int, optimal_time=None) -> dict:
        config = BACKOFF_RULES.get(cause, BACKOFF_RULES["insufficient_funds"])

        if attempt >= config.max_attempts:
            return {"exhausted": True, "schedule_at": None}

        if optimal_time and cause == "insufficient_funds":
            jitter = timedelta(minutes=random.randint(-15, 15))
            return {"exhausted": False, "schedule_at": optimal_time + jitter, "strategy": "payday_inference"}

        base_delay = config.base_hours * (2 ** attempt)
        jitter_min = random.randint(0, config.jitter_minutes)
        schedule_at = datetime.now() + timedelta(hours=base_delay, minutes=jitter_min)

        return {
            "exhausted": False, "schedule_at": schedule_at,
            "strategy": "exponential_backoff", "delay_hours": round(base_delay, 2),
        }
