"""crai/agent/state.py — Schema de estado do agente (churn involuntário)."""

from typing import TypedDict, Optional
from datetime import datetime


class AgentState(TypedDict):
    # Input
    stripe_event:   dict
    customer_id:    str
    amount:         float
    invoice_id:     str

    # Diagnóstico (XGBoost)
    failure_cause:        Optional[str]
    recovery_score:        Optional[float]
    feature_importance:    Optional[dict]

    # Anomalia (Autoencoder)
    is_anomalous:         Optional[bool]
    reconstruction_error: Optional[float]

    # Liquidez (LSTM + Prophet)
    optimal_retry_at: Optional[datetime]
    confidence:       Optional[float]
    profile_type:     Optional[str]

    # Retentativa (Backoff)
    retry_count:     int
    next_retry_at:   Optional[datetime]
    retry_exhausted: bool
    recovered:       bool

    # Dunning (LangGraph + Claude)
    dunning_sent: bool
    channel:      Optional[str]
    message_sent: Optional[str]
