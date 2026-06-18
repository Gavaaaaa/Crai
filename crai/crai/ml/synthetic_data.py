"""
crai/ml/synthetic_data.py — Gerador de dataset sintético para treino do failure_classifier.

Gera dados plausíveis para o mercado brasileiro de SaaS B2B:
- Tenure com distribuição realista (concentração em 0-12 meses, cauda longa até 60+)
- Sazonalidade de pagamento brasileira (dias 5, 10, 15, 30)
- Códigos de falha de gateway com distribuição desbalanceada
- LTV estimado para cálculo do e-Profit
- Seed fixa (42) para reprodutibilidade
"""

import numpy as np
import pandas as pd
from typing import Optional

# Seed global para reprodutibilidade
SEED = 42

# Códigos de falha de gateway comuns no mercado brasileiro
GATEWAY_ERROR_CODES = [
    "insufficient_funds",
    "expired_card",
    "card_declined",
    "processing_error",
    "do_not_honor",
    "generic_decline",
]

# Probabilidades de cada código (distribuição desbalanceada — saldo insuficiente domina)
ERROR_CODE_PROBS = [0.35, 0.20, 0.15, 0.10, 0.12, 0.08]

# Bandeiras de cartão comuns no Brasil
CARD_BRANDS = ["visa", "mastercard", "elo", "amex", "hipercard"]
CARD_BRAND_PROBS = [0.40, 0.30, 0.15, 0.08, 0.07]


def generate_dataset(n_samples: int = 3000, seed: Optional[int] = SEED) -> pd.DataFrame:
    """
    Gera dataset sintético com distribuições plausíveis para SaaS B2B brasileiro.

    Args:
        n_samples: Número de amostras (default 3000)
        seed: Seed para reprodutibilidade (default 42)

    Returns:
        DataFrame com features e target 'recovered'
    """
    rng = np.random.default_rng(seed)

    # ── Tenure (meses de casa) ───────────────────────────────────────
    # Distribuição realista: ~60% em 0-12 meses, cauda longa até 60+
    # Mistura de exponencial (clientes novos) + uniforme (clientes antigos)
    tenure_new = rng.exponential(scale=6.0, size=int(n_samples * 0.65))
    tenure_old = rng.uniform(12, 72, size=int(n_samples * 0.35))
    tenure_all = np.concatenate([tenure_new, tenure_old])
    rng.shuffle(tenure_all)
    tenure_months = np.clip(tenure_all[:n_samples], 0, 72).astype(int)

    # ── Dia do mês da cobrança ───────────────────────────────────────
    # Sazonalidade brasileira: concentração nos dias 5, 10, 15, 30
    peak_days = [5, 10, 15, 20, 30]
    day_of_month = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if rng.random() < 0.6:  # 60% nos dias de pico
            day_of_month[i] = rng.choice(peak_days)
        else:
            day_of_month[i] = rng.integers(1, 29)

    # ── Valor da fatura (LogNormal) ──────────────────────────────────
    # SaaS B2B brasileiro: R$99 a R$5000, concentração em R$200-R$800
    invoice_amount = np.clip(
        rng.lognormal(mean=5.8, sigma=0.7, size=n_samples),
        49.90, 9999.90
    ).round(2)

    # ── Ticket médio (correlacionado com fatura, com ruído) ──────────
    avg_ticket = (invoice_amount * rng.uniform(0.85, 1.15, size=n_samples)).round(2)

    # ── Código de erro do gateway ────────────────────────────────────
    gateway_error_code = rng.choice(
        GATEWAY_ERROR_CODES, size=n_samples, p=ERROR_CODE_PROBS
    )

    # ── Bandeira do cartão ───────────────────────────────────────────
    card_brand = rng.choice(CARD_BRANDS, size=n_samples, p=CARD_BRAND_PROBS)

    # ── Histórico de pagamento (score 0-1) ───────────────────────────
    # Correlação positiva com tenure (clientes antigos tendem a ter histórico melhor)
    tenure_factor = np.clip(tenure_months / 60, 0, 1)
    payment_history_score = np.clip(
        rng.beta(5, 2, size=n_samples) * 0.7 + tenure_factor * 0.3,
        0, 1
    ).round(3)

    # ── Número de falhas nos últimos 90 dias ─────────────────────────
    failure_count_90d = rng.poisson(lam=1.5, size=n_samples)

    # ── Hora da tentativa de cobrança ────────────────────────────────
    hour_of_day = rng.choice(
        range(24), size=n_samples,
        p=_hour_distribution()
    )

    # ── Dia da semana (0=seg, 6=dom) ─────────────────────────────────
    day_of_week = rng.integers(0, 7, size=n_samples)

    # ── Número de tentativas anteriores ──────────────────────────────
    attempt_count = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.45, 0.30, 0.15, 0.10])

    # ── LTV estimado ─────────────────────────────────────────────────
    # LTV = tenure * avg_ticket_mensal * fator_retenção
    retention_factor = np.clip(0.85 + tenure_factor * 0.10, 0.80, 0.98)
    ltv_estimated = (tenure_months * avg_ticket * retention_factor / 12).round(2)
    # Mínimo de LTV = valor da fatura (pelo menos 1 mês)
    ltv_estimated = np.maximum(ltv_estimated, invoice_amount).round(2)

    # ══ TARGET: recovered (0/1) ══════════════════════════════════════
    # Probabilidade de recuperação baseada em fatores realistas
    p_recovery = _calculate_recovery_probability(
        tenure_months=tenure_months,
        payment_history_score=payment_history_score,
        gateway_error_code=gateway_error_code,
        invoice_amount=invoice_amount,
        failure_count_90d=failure_count_90d,
        day_of_month=day_of_month,
        attempt_count=attempt_count,
        rng=rng,
    )
    recovered = (rng.random(n_samples) < p_recovery).astype(int)

    # ── Montar DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "day_of_month": day_of_month,
        "invoice_amount": invoice_amount,
        "avg_ticket": avg_ticket,
        "gateway_error_code": gateway_error_code,
        "card_brand": card_brand,
        "payment_history_score": payment_history_score,
        "failure_count_90d": failure_count_90d,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "attempt_count": attempt_count,
        "ltv_estimated": ltv_estimated,
        "recovered": recovered,
    })

    return df


def _calculate_recovery_probability(
    tenure_months: np.ndarray,
    payment_history_score: np.ndarray,
    gateway_error_code: np.ndarray,
    invoice_amount: np.ndarray,
    failure_count_90d: np.ndarray,
    day_of_month: np.ndarray,
    attempt_count: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Calcula probabilidade de recuperação com base em fatores realistas.
    Modela as correlações que o XGBoost/RF devem aprender.
    """
    n = len(tenure_months)
    p = np.full(n, 0.5)

    # Tenure: correlação negativa forte com churn (clientes antigos recuperam mais)
    p += np.clip(tenure_months / 60, 0, 0.25)

    # Histórico de pagamento: bom histórico → maior recuperação
    p += (payment_history_score - 0.5) * 0.3

    # Código de erro: impacto diferente por tipo
    error_impact = {
        "insufficient_funds": -0.05,   # Recuperável com retry no payday
        "expired_card": -0.15,         # Precisa atualizar cartão
        "card_declined": -0.25,        # Bloqueio bancário — difícil
        "processing_error": 0.05,      # Erro técnico — retry geralmente resolve
        "do_not_honor": -0.30,         # Banco recusou — muito difícil
        "generic_decline": -0.20,      # Incerto
    }
    for code, impact in error_impact.items():
        mask = gateway_error_code == code
        p[mask] += impact

    # Valor da fatura: faturas muito altas → menor recuperação
    p -= np.clip((invoice_amount - 500) / 5000, 0, 0.15)

    # Falhas recentes: muitas falhas → menor recuperação
    p -= np.clip(failure_count_90d * 0.05, 0, 0.20)

    # Dia do mês: dias de pagamento (5, 10, 15) → melhor recuperação
    payday_mask = np.isin(day_of_month, [5, 6, 7, 10, 15, 20, 30])
    p[payday_mask] += 0.08

    # Tentativas anteriores: mais tentativas → menor chance
    p -= (attempt_count - 1) * 0.06

    # Ruído aleatório para evitar separação perfeita
    p += rng.normal(0, 0.05, size=n)

    return np.clip(p, 0.05, 0.95)


def _hour_distribution() -> list:
    """Distribuição de tentativas de cobrança por hora (concentração em horário comercial)."""
    probs = np.zeros(24)
    # Madrugada: baixo
    probs[0:6] = 0.5
    # Manhã: alto (processamento batch dos gateways)
    probs[6:12] = 3.0
    # Tarde: médio-alto
    probs[12:18] = 2.5
    # Noite: médio
    probs[18:24] = 1.5
    # Normalizar
    probs = probs / probs.sum()
    return probs.tolist()


if __name__ == "__main__":
    # Gerar e inspecionar dataset
    df = generate_dataset(3000)
    print(f"Dataset gerado: {df.shape}")
    print(f"\nDistribuição do target:")
    print(df["recovered"].value_counts(normalize=True).round(3))
    print(f"\nTenure (meses):")
    print(df["tenure_months"].describe().round(1))
    print(f"\nCódigos de erro:")
    print(df["gateway_error_code"].value_counts(normalize=True).round(3))
    print(f"\nLTV estimado:")
    print(df["ltv_estimated"].describe().round(1))
    print(f"\nTaxa de recuperação por código de erro:")
    print(df.groupby("gateway_error_code")["recovered"].mean().round(3))
    print(f"\nTaxa de recuperação por faixa de tenure:")
    bins = [0, 3, 6, 12, 24, 72]
    labels = ["0-3m", "3-6m", "6-12m", "12-24m", "24+m"]
    df["tenure_bin"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)
    print(df.groupby("tenure_bin")["recovered"].mean().round(3))
