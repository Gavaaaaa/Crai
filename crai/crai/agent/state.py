"""crai/agent/state.py — Schema de estado do agente (churn involuntário)."""

from typing import TypedDict, Optional
from datetime import datetime


class AgentState(TypedDict):
    # Input
    stripe_event:   dict
    customer_id:    str
    amount:         float
    invoice_id:     str

    # Diagnóstico (XGBoost + RF + e-Profit + SHAP)
    failure_cause:        Optional[str]
    recovery_score:       Optional[int]           # 0-100
    p_recovery:           Optional[float]          # 0.0-1.0
    eprofit:              Optional[float]          # R$ — e-Profit da intervenção
    recommend_action:     Optional[bool]           # True se e-Profit > 0
    ltv_estimated:        Optional[float]          # LTV estimado do cliente
    shap_explanation:     Optional[dict]           # Explicação SHAP por feature
    feature_importance:   Optional[dict]

    # Anomalia (Autoencoder)
    is_anomalous:         Optional[bool]
    reconstruction_error: Optional[float]
    anomaly_explanation:  Optional[list]           # Top features por erro de reconstrução

    # Liquidez (LSTM + Prophet)
    optimal_retry_at: Optional[datetime]
    confidence:       Optional[float]
    profile_type:     Optional[str]

    # Decisão de recuperação (Módulo 5 — raciocínio ReAct sobre contexto ML)
    estrategia:   Optional[str]    # retry_automatico | mensagem_pagamento
    raciocinio:   Optional[list]   # trilha de raciocínio em PT-BR (auditoria)

    # Retentativa (Backoff)
    retry_count:     int
    next_retry_at:   Optional[datetime]
    retry_exhausted: bool
    recovered:       bool

    # Dunning (LangGraph + Claude) — mensagem personalizada, sem escalonamento humano
    dunning_sent:     bool
    channel:          Optional[str]
    metodo_pagamento: Optional[str]   # pix_automatico | boleto (Pix Automático como fallback antes do boleto)
    message_sent:     Optional[str]
