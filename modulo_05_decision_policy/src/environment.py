"""Ambiente simulado de cobranca (dunning) para o Modulo 5 do CRAI.

Define a "verdade" que a politica de decisao NAO conhece: por que cada
cobranca falhou de verdade, quando o cliente volta a ter saldo e qual a
chance de ele agir em cada canal.

O ponto central do ambiente e que RETENTAR nao resolve todas as causas:

- insufficient_funds : o retry so passa DEPOIS que o cliente recebe. Antes
                       disso, toda retentativa e dinheiro jogado fora.
- expired_card       : o retry NUNCA passa. O cartao esta vencido; so o
                       cliente pode resolver, e para isso precisa ser
                       contatado por algum canal.
- processing_error   : falha transitoria do gateway; o retry resolve quase
                       sempre, e imediatamente.

E exatamente essa assimetria que a regra fixa de mercado (retentar em dias
fixos, ignorando a causa) desperdica — e que a politica orientada a
e-Profit explora.
"""
from __future__ import annotations

import numpy as np

# ── Causas de falha (mesma distribuicao do Modulo 1) ─────────────────────
CAUSAS = ["insufficient_funds", "expired_card", "card_declined",
          "processing_error", "do_not_honor", "generic_decline"]
CAUSA_PROBS = [0.35, 0.20, 0.15, 0.10, 0.12, 0.08]

PERFIS = ["CLT", "PJ", "freelancer"]
PERFIL_PROBS = [0.50, 0.30, 0.20]

# ── Acoes disponiveis e custo unitario em R$ ─────────────────────────────
# Mesmos custos de crai/ml/failure_classifier.INTERVENTION_COSTS.
ACOES = ["retry", "email", "whatsapp", "pix_boleto", "ligacao_cs", "desistir"]
CUSTO_ACAO = {
    "retry":      0.02,   # taxa do gateway por tentativa
    "email":      0.02,
    "whatsapp":   0.05,
    "pix_boleto": 0.50,
    "ligacao_cs": 15.00,
    "desistir":   0.00,
}
CANAIS = ["email", "whatsapp", "pix_boleto", "ligacao_cs"]

HORIZONTE_DIAS = 14   # janela de cobranca antes de dar a fatura por perdida
MAX_ACOES = 4         # orcamento de intervencoes por cobranca

# ══ VERDADE OCULTA ═══════════════════════════════════════════════════════

# P(o retry sozinho resolve | causa), assumindo que ha saldo disponivel.
RETRY_RESOLVE = {
    "insufficient_funds": 0.80,
    "processing_error":   0.90,
    "generic_decline":    0.25,
    "card_declined":      0.15,
    "do_not_honor":       0.05,
    "expired_card":       0.00,   # cartao vencido nao passa, por mais que se tente
}

# P(o cliente age | canal) — abrir a mensagem e efetivamente concluir o pagamento.
RESPOSTA_CANAL = {"email": 0.25, "whatsapp": 0.45, "pix_boleto": 0.55, "ligacao_cs": 0.75}

# P(o cliente CONSEGUE pagar | causa) — teto de recuperabilidade via acao do cliente.
CAPACIDADE = {
    "insufficient_funds": 0.85,
    "expired_card":       0.90,   # basta atualizar o cartao
    "processing_error":   0.95,
    "generic_decline":    0.60,
    "card_declined":      0.50,
    "do_not_honor":       0.35,
}

# Ancoras de pagamento por perfil (dia do mes tipico de entrada de dinheiro).
ANCORAS_PAYDAY = {"CLT": [5, 20], "PJ": [10, 15, 28], "freelancer": [3, 8, 14, 22]}


def custo(acao: str) -> float:
    return CUSTO_ACAO[acao]


class DunningEnvironment:
    """Sorteia cobrancas falhas e responde se cada acao recuperou o pagamento."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    # ── Geracao de cobrancas ─────────────────────────────────────────────
    def sample_cobranca(self) -> dict:
        causa = str(self.rng.choice(CAUSAS, p=CAUSA_PROBS))
        perfil = str(self.rng.choice(PERFIS, p=PERFIL_PROBS))

        valor = float(np.clip(self.rng.lognormal(5.8, 0.7), 49.90, 9999.90))
        tenure = int(np.clip(self.rng.exponential(12), 0, 72))
        ltv = round(max(valor, tenure * valor * 0.9 / 12), 2)

        # Dia (0..13) em que o cliente volta a ter saldo. So restringe o
        # retry na causa insufficient_funds; nas demais o dinheiro existe.
        if causa == "insufficient_funds":
            dia_liquidez = int(self.rng.choice(ANCORAS_PAYDAY[perfil])) % HORIZONTE_DIAS
        else:
            dia_liquidez = 0

        return {
            "causa": causa,
            "perfil": perfil,
            "valor": round(valor, 2),
            "ltv": ltv,
            "tenure": tenure,
            "dia_liquidez": dia_liquidez,
            # Sorte pre-sorteada, um valor por passo. Garante que todas as
            # politicas enfrentem exatamente o mesmo acaso (comparacao pareada)
            # mesmo tomando numeros diferentes de acoes.
            "_u": self.rng.uniform(size=MAX_ACOES).tolist(),
        }

    # ── Resposta do ambiente a uma acao ──────────────────────────────────
    def responder(self, cobranca: dict, acao: str, dia: int, passo: int) -> bool:
        """A acao recuperou o pagamento?"""
        if acao == "desistir":
            return False

        causa = cobranca["causa"]
        tem_saldo = dia >= cobranca["dia_liquidez"]
        u = cobranca["_u"][passo]

        if acao == "retry":
            # Retentar sem saldo e desperdicio garantido.
            if not tem_saldo:
                return False
            return bool(u < RETRY_RESOLVE[causa])

        # Canais dependem do cliente agir E de ele ter como pagar.
        if not tem_saldo:
            return False
        return bool(u < RESPOSTA_CANAL[acao] * CAPACIDADE[causa])

    # ── Contexto observavel pela politica (saida dos Modulos 1-3) ────────
    def contexto_ml(self, cobranca: dict, rng: np.random.Generator) -> dict:
        """O que a politica ENXERGA — nao a verdade, mas a estimativa dos modelos.

        - causa vem do diagnostico do Modulo 1 (assumido correto: o codigo do
          gateway e observavel);
        - p_recovery e uma estimativa ruidosa da recuperabilidade real;
        - payday_previsto carrega o erro do Modulo 3 (MAE ~0.6 dia) contra o
          erro da heuristica de dias fixos que ele substituiu (MAE ~5 dias).
        """
        causa = cobranca["causa"]
        p_real = CAPACIDADE[causa]
        p_estimado = float(np.clip(p_real + rng.normal(0, 0.08), 0.05, 0.95))

        dia_real = cobranca["dia_liquidez"]
        payday_crai = int(np.clip(round(dia_real + rng.normal(0, 0.6)), 0, HORIZONTE_DIAS - 1))
        payday_heur = int(np.clip(round(dia_real + rng.normal(0, 5.0)), 0, HORIZONTE_DIAS - 1))

        return {
            "causa": causa,
            "perfil": cobranca["perfil"],
            "valor": cobranca["valor"],
            "ltv": cobranca["ltv"],
            "p_recovery": round(p_estimado, 4),
            "payday_previsto": payday_crai,
            "payday_heuristico": payday_heur,
        }
