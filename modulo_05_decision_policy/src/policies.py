"""Politicas de decisao comparadas no Modulo 5.

Todas expoem a mesma interface: dada a cobranca em aberto, decidem QUAL acao
tomar e EM QUE DIA — ou encerram o ciclo. A diferenca esta em quanta
informacao cada uma usa.

- RegraFixa      : baseline de mercado. Retenta em dias fixos e manda um
                   e-mail no fim. Nao olha causa, valor, LTV nem payday.
- SempreLigacao  : maximiza recuperacao ignorando custo. Existe para mostrar
                   que "recuperar mais" e "lucrar mais" sao objetivos
                   diferentes — o mesmo argumento do Modulo 4.
- Crai           : escolhe a acao de maior e-Profit esperado usando o
                   contexto dos Modulos 1-3 (causa, recuperabilidade, valor,
                   payday previsto). Raciona o time de CS por ganho marginal.
- Oraculo        : conhece a verdade oculta do ambiente. Teto pratico.

O time de CS e um recurso ESCASSO: `orcamento_cs` limita quantas ligacoes a
politica pode fazer em toda a simulacao. E isso que torna a decisao
interessante — nao basta saber que a ligacao converte mais, e preciso
decidir QUEM recebe a ligacao.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.environment import (CANAIS, CAPACIDADE, CUSTO_ACAO, HORIZONTE_DIAS,
                             RESPOSTA_CANAL, RETRY_RESOLVE)

# ── Crencas da politica CRAI (priors declarados, nao a verdade oculta) ────
# Analogo ao SEED_PRIORS do Modulo 4: valores de benchmark que em producao
# seriam substituidos por taxas observadas.
CRENCA_RETRY = {
    "insufficient_funds": 0.75,
    "processing_error":   0.85,
    "generic_decline":    0.25,
    "card_declined":      0.15,
    "do_not_honor":       0.05,
    "expired_card":       0.00,
}
CRENCA_RESPOSTA = {"email": 0.25, "whatsapp": 0.45, "pix_boleto": 0.55, "ligacao_cs": 0.75}

# Ganho marginal minimo (R$) para gastar uma ligacao humana em vez do melhor
# canal automatico. Raciona o time de CS para os casos de maior valor.
LIMIAR_MARGINAL_CS = 200.0


class Policy(ABC):
    nome: str = "policy"

    def reset(self, orcamento_cs: int) -> None:
        self.orcamento_cs = orcamento_cs

    def _pode_ligar(self) -> bool:
        return self.orcamento_cs > 0

    def _consumir_cs(self) -> None:
        self.orcamento_cs -= 1

    @abstractmethod
    def decidir(self, ctx: dict, passo: int, dia_atual: int) -> tuple[int, str] | None:
        """Retorna (dia_da_acao, acao) ou None para encerrar o ciclo."""


class RegraFixaPolicy(Policy):
    """Baseline de mercado: 3 retentativas em dias fixos + 1 e-mail."""

    nome = "regra_fixa"
    DIAS_RETRY = [1, 3, 5]
    DIA_EMAIL = 7

    def decidir(self, ctx, passo, dia_atual):
        if passo < len(self.DIAS_RETRY):
            return self.DIAS_RETRY[passo], "retry"
        if passo == len(self.DIAS_RETRY):
            return self.DIA_EMAIL, "email"
        return None


class SempreLigacaoPolicy(Policy):
    """Maximiza recuperacao: liga sempre que houver time disponivel."""

    nome = "sempre_ligacao"

    def decidir(self, ctx, passo, dia_atual):
        if passo == 0 and self._pode_ligar():
            self._consumir_cs()
            return 1, "ligacao_cs"
        if passo <= 2:
            return max(dia_atual + 1, 1 + passo * 3), "whatsapp"
        return None


class CraiPolicy(Policy):
    """Escolhe a acao de maior e-Profit esperado com o contexto dos Modulos 1-3."""

    nome = "crai_eprofit"

    def __init__(self, usar_payday_ml: bool = True) -> None:
        self.usar_payday_ml = usar_payday_ml

    def decidir(self, ctx, passo, dia_atual):
        if passo >= 3:
            return None

        causa = ctx["causa"]
        valor = ctx["valor"]
        p = ctx["p_recovery"]

        chave_payday = "payday_previsto" if self.usar_payday_ml else "payday_heuristico"
        dia_saldo = ctx[chave_payday] if causa == "insufficient_funds" else 0

        # ── Passo 0: retry so quando a causa e mecanicamente retentavel ──
        if passo == 0:
            ep_retry = CRENCA_RETRY[causa] * valor - CUSTO_ACAO["retry"]
            melhor_canal, ep_canal = self._melhor_canal(p, valor, permitir_cs=False)
            if ep_retry >= ep_canal and ep_retry > 0:
                # Retentar antes do saldo entrar e desperdicio garantido.
                return max(dia_saldo, dia_atual + 1), "retry"
            return self._agir_em_canal(ctx, max(dia_saldo, dia_atual + 1), p, valor)

        # ── Passos seguintes: o retry ja falhou, contatar o cliente ──────
        dia = max(dia_saldo, dia_atual + 1)
        if dia >= HORIZONTE_DIAS:
            return None
        return self._agir_em_canal(ctx, dia, p, valor)

    # ── Helpers ──────────────────────────────────────────────────────────
    def _melhor_canal(self, p, valor, permitir_cs):
        candidatos = CANAIS if permitir_cs else [c for c in CANAIS if c != "ligacao_cs"]
        vals = {c: p * CRENCA_RESPOSTA[c] * valor - CUSTO_ACAO[c] for c in candidatos}
        melhor = max(vals, key=vals.get)
        return melhor, vals[melhor]

    def _agir_em_canal(self, ctx, dia, p, valor):
        auto, ep_auto = self._melhor_canal(p, valor, permitir_cs=False)
        ep_cs = p * CRENCA_RESPOSTA["ligacao_cs"] * valor - CUSTO_ACAO["ligacao_cs"]

        # Escala para humano so quando o ganho marginal justifica o recurso escasso.
        if self._pode_ligar() and (ep_cs - ep_auto) > LIMIAR_MARGINAL_CS:
            self._consumir_cs()
            return dia, "ligacao_cs"
        if ep_auto <= 0:
            return None          # nao vale intervir — aqui o gate de e-Profit morde
        return dia, auto


class OraclePolicy(Policy):
    """Teto pratico: conhece dia de liquidez e probabilidades verdadeiras."""

    nome = "oraculo"

    def decidir(self, ctx, passo, dia_atual):
        if passo >= 3:
            return None
        causa = ctx["causa"]
        valor = ctx["valor"]
        p_real = CAPACIDADE[causa]
        dia_saldo = ctx["_dia_liquidez_real"]
        dia = max(dia_saldo, dia_atual + 1)
        if dia >= HORIZONTE_DIAS:
            return None

        if passo == 0:
            ep_retry = RETRY_RESOLVE[causa] * valor - CUSTO_ACAO["retry"]
            vals = {c: p_real * RESPOSTA_CANAL[c] * valor - CUSTO_ACAO[c]
                    for c in CANAIS if c != "ligacao_cs"}
            if ep_retry >= max(vals.values()):
                return dia, "retry"

        vals = {c: p_real * RESPOSTA_CANAL[c] * valor - CUSTO_ACAO[c]
                for c in CANAIS if c != "ligacao_cs"}
        auto = max(vals, key=vals.get)
        ep_cs = p_real * RESPOSTA_CANAL["ligacao_cs"] * valor - CUSTO_ACAO["ligacao_cs"]
        if self._pode_ligar() and (ep_cs - vals[auto]) > LIMIAR_MARGINAL_CS:
            self._consumir_cs()
            return dia, "ligacao_cs"
        if vals[auto] <= 0:
            return None
        return dia, auto


def todas_politicas() -> list[Policy]:
    return [RegraFixaPolicy(), SempreLigacaoPolicy(), CraiPolicy(), OraclePolicy()]
