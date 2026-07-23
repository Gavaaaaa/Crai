"""Simulacao do Modulo 5: politica orientada a e-Profit vs regra fixa.

Roda as 4 politicas sobre EXATAMENTE o mesmo fluxo de cobrancas falhas e com
a mesma sorte pre-sorteada por cobranca (comparacao pareada). Cada politica
tem o mesmo orcamento de ligacoes humanas.

Ciclo de uma cobranca: ate MAX_ACOES intervencoes dentro de HORIZONTE_DIAS
dias. Para na primeira que recuperar o pagamento ou quando a politica
encerra. Metrica principal:

    receita liquida = faturas recuperadas - custo das intervencoes

Salva:
- reports/simulacao.csv : uma linha por (politica, cobranca)
- models/meta.json      : parametros da simulacao
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment import (CUSTO_ACAO, HORIZONTE_DIAS, MAX_ACOES,
                             DunningEnvironment)
from src.policies import LIMIAR_MARGINAL_CS, todas_politicas

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Fracao do volume que o time de CS consegue atender.
CAPACIDADE_CS = 0.10


def _rodar_cobranca(politica, env, cobranca, ctx) -> dict:
    """Executa o ciclo de cobranca de uma fatura sob uma politica."""
    dia_atual = 0
    custo_total = 0.0
    acoes: list[str] = []
    recuperou = False

    for passo in range(MAX_ACOES):
        decisao = politica.decidir(ctx, passo, dia_atual)
        if decisao is None:
            break
        dia, acao = decisao
        if dia >= HORIZONTE_DIAS:
            break

        dia_atual = dia
        custo_total += CUSTO_ACAO[acao]
        acoes.append(acao)

        if env.responder(cobranca, acao, dia, passo):
            recuperou = True
            break

    return {
        "recuperou": int(recuperou),
        "receita": cobranca["valor"] if recuperou else 0.0,
        "custo": round(custo_total, 2),
        "n_acoes": len(acoes),
        "acoes": "|".join(acoes),
        "usou_cs": int("ligacao_cs" in acoes),
        "retries_desperdicados": sum(
            1 for i, a in enumerate(acoes)
            if a == "retry" and not recuperou
        ),
    }


def simular(n_cobrancas: int = 5000, seed: int = 42) -> dict:
    # Fluxo unico de cobrancas, identico para todas as politicas.
    env_gen = DunningEnvironment(seed=seed)
    cobrancas = [env_gen.sample_cobranca() for _ in range(n_cobrancas)]

    # Contexto ML observavel (mesmo ruido de estimativa para todas as politicas)
    rng_ctx = np.random.default_rng(seed + 7)
    contextos = [env_gen.contexto_ml(c, rng_ctx) for c in cobrancas]
    for ctx, cob in zip(contextos, cobrancas):
        ctx["_dia_liquidez_real"] = cob["dia_liquidez"]   # so o oraculo consulta

    orcamento = int(n_cobrancas * CAPACIDADE_CS)
    env = DunningEnvironment(seed=seed)   # respostas vem do _u pre-sorteado

    linhas = []
    for politica in todas_politicas():
        politica.reset(orcamento_cs=orcamento)
        for i, (cobranca, ctx) in enumerate(zip(cobrancas, contextos)):
            r = _rodar_cobranca(politica, env, cobranca, ctx)
            linhas.append({
                "politica": politica.nome, "cobranca": i,
                "causa": cobranca["causa"], "perfil": cobranca["perfil"],
                "valor": cobranca["valor"], "ltv": cobranca["ltv"],
                **r,
            })

    df = pd.DataFrame(linhas)
    df["liquido"] = (df["receita"] - df["custo"]).round(2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "simulacao.csv", index=False)

    meta = {
        "n_cobrancas": n_cobrancas,
        "seed": seed,
        "horizonte_dias": HORIZONTE_DIAS,
        "max_acoes": MAX_ACOES,
        "capacidade_cs": CAPACIDADE_CS,
        "orcamento_cs": orcamento,
        "limiar_marginal_cs": LIMIAR_MARGINAL_CS,
        "politicas": [p.nome for p in todas_politicas()],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    resumo = (
        df.groupby("politica")
        .agg(receita=("receita", "sum"), custo=("custo", "sum"),
             liquido=("liquido", "sum"), taxa_recuperacao=("recuperou", "mean"),
             acoes_por_cobranca=("n_acoes", "mean"))
        .round(2)
        .sort_values("liquido")
    )
    print(f"[ok] {n_cobrancas} cobrancas x {len(meta['politicas'])} politicas "
          f"| orcamento CS = {orcamento} ligacoes")
    print(resumo.to_string())
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cobrancas", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simular(args.cobrancas, args.seed)


if __name__ == "__main__":
    main()
