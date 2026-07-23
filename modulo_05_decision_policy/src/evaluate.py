"""Avaliacao do Modulo 5 — consolida reports/metricas.json.

Alem do resumo por politica, responde tres perguntas que a banca faz:

1. Quanto do teto do oraculo a politica CRAI alcanca?
2. Onde exatamente a regra fixa perde dinheiro? (recuperacao por causa e
   retentativas desperdicadas)
3. Quanto do ganho vem do Modulo 3? (ablacao: mesma politica com o payday
   previsto pela LSTM+Prophet vs com a heuristica de dias fixos que ele
   substituiu)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.environment import DunningEnvironment
from src.policies import CraiPolicy
from src.simulate import CAPACIDADE_CS, _rodar_cobranca

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def _resumo_por_politica(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("politica")
        .agg(receita=("receita", "sum"),
             custo=("custo", "sum"),
             liquido=("liquido", "sum"),
             taxa_recuperacao=("recuperou", "mean"),
             acoes_por_cobranca=("n_acoes", "mean"),
             retries_desperdicados=("retries_desperdicados", "sum"),
             ligacoes=("usou_cs", "sum"))
        .round(4)
    )


def _ablacao_payday(n_cobrancas: int, seed: int) -> dict:
    """Mesma politica CRAI, trocando so a fonte da previsao de payday."""
    env_gen = DunningEnvironment(seed=seed)
    cobrancas = [env_gen.sample_cobranca() for _ in range(n_cobrancas)]
    rng_ctx = np.random.default_rng(seed + 7)
    contextos = [env_gen.contexto_ml(c, rng_ctx) for c in cobrancas]
    for ctx, cob in zip(contextos, cobrancas):
        ctx["_dia_liquidez_real"] = cob["dia_liquidez"]

    env = DunningEnvironment(seed=seed)
    orcamento = int(n_cobrancas * CAPACIDADE_CS)

    out = {}
    for rotulo, usar_ml in [("payday_lstm_prophet", True), ("payday_heuristico", False)]:
        pol = CraiPolicy(usar_payday_ml=usar_ml)
        pol.reset(orcamento_cs=orcamento)
        liquido = 0.0
        recuperadas = 0
        for cobranca, ctx in zip(cobrancas, contextos):
            r = _rodar_cobranca(pol, env, cobranca, ctx)
            liquido += r["receita"] - r["custo"]
            recuperadas += r["recuperou"]
        out[rotulo] = {
            "liquido": round(liquido, 2),
            "taxa_recuperacao": round(recuperadas / n_cobrancas, 4),
        }
    out["ganho_do_modulo_3"] = round(
        out["payday_lstm_prophet"]["liquido"] - out["payday_heuristico"]["liquido"], 2
    )
    return out


def avaliar() -> dict:
    df = pd.read_csv(REPORTS_DIR / "simulacao.csv")
    n = df["cobranca"].nunique()

    resumo = _resumo_por_politica(df)
    base = float(resumo.loc["regra_fixa", "liquido"])
    crai = float(resumo.loc["crai_eprofit", "liquido"])
    teto = float(resumo.loc["oraculo", "liquido"])

    # Recuperacao por causa: mostra ONDE a regra fixa perde
    por_causa = (
        df.pivot_table(index="causa", columns="politica",
                       values="recuperou", aggfunc="mean")
        .round(4)
    )

    metricas = {
        "n_cobrancas": int(n),
        "volume_total_em_risco": round(float(df[df.politica == "regra_fixa"]["valor"].sum()), 2),
        "por_politica": json.loads(resumo.to_json(orient="index")),
        "ganho_crai_vs_regra_fixa": round(crai - base, 2),
        "ganho_pct": round((crai / base - 1) * 100, 2),
        "ganho_por_cobranca": round((crai - base) / n, 2),
        "pct_do_teto_do_oraculo": round(crai / teto * 100, 2),
        "recuperacao_por_causa": json.loads(por_causa.to_json(orient="index")),
        "ablacao_payday": _ablacao_payday(n, seed=42),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(resumo.to_string())
    print(f"\nGanho CRAI vs regra fixa : R$ {metricas['ganho_crai_vs_regra_fixa']:,.2f} "
          f"({metricas['ganho_pct']:+.1f}%)  |  R$ {metricas['ganho_por_cobranca']:.2f}/cobranca")
    print(f"% do teto do oraculo     : {metricas['pct_do_teto_do_oraculo']:.1f}%")
    print(f"\nTaxa de recuperacao por causa:\n{por_causa.to_string()}")
    abl = metricas["ablacao_payday"]
    print(f"\nAblacao (contribuicao do Modulo 3): R$ {abl['ganho_do_modulo_3']:,.2f}")
    print(f"  com LSTM+Prophet : R$ {abl['payday_lstm_prophet']['liquido']:,.2f} "
          f"| recup. {abl['payday_lstm_prophet']['taxa_recuperacao']:.1%}")
    print(f"  com heuristica   : R$ {abl['payday_heuristico']['liquido']:,.2f} "
          f"| recup. {abl['payday_heuristico']['taxa_recuperacao']:.1%}")
    print(f"\n[ok] metricas salvas em {REPORTS_DIR / 'metricas.json'}")
    return metricas


if __name__ == "__main__":
    avaliar()
