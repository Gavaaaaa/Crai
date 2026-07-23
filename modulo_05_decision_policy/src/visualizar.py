"""Figuras do Modulo 5 para o relatorio da banca.

Gera reports/figures/*.png. Paleta categorica validada para daltonismo
(azul #2a78d6 / laranja #eb6834: CVD dE 24.7, visao normal dE 33.6, ambos
acima dos pisos; contraste >= 3:1 na superficie clara).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# ── Tokens de cor (superficie clara) ─────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIE_1 = "#2a78d6"   # azul  — CRAI / medida unica
SERIE_2 = "#eb6834"   # laranja — regra fixa

ROTULOS = {
    "regra_fixa": "Regra fixa (mercado)",
    "sempre_ligacao": "Sempre ligacao",
    "crai_eprofit": "CRAI (e-Profit)",
    "oraculo": "Oraculo (teto)",
}
ROTULO_CAUSA = {
    "insufficient_funds": "Saldo insuficiente",
    "expired_card": "Cartao expirado",
    "card_declined": "Cartao recusado",
    "processing_error": "Erro tecnico",
    "do_not_honor": "Banco nao honrou",
    "generic_decline": "Recusa generica",
}


def _base(ax) -> None:
    """Cromo recessivo: sem molduras, grade hairline, eixos em tom mudo."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(BASELINE)
        ax.spines[lado].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_axisbelow(True)


def fig_liquido(df: pd.DataFrame) -> None:
    resumo = df.groupby("politica")["liquido"].sum().sort_values()
    desperdicio = df.groupby("politica")["retries_desperdicados"].sum()

    fig, ax = plt.subplots(figsize=(8.2, 4.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    nomes = [ROTULOS[p] for p in resumo.index]
    ax.barh(nomes, resumo.values / 1000, color=SERIE_1, height=0.62)

    for i, (pol, val) in enumerate(resumo.items()):
        rotulo = f"R$ {val/1000:,.0f}k".replace(",", ".")
        ax.text(val / 1000 + 18, i, rotulo, va="center", fontsize=9.5, color=INK)
        # So o contraste que importa: o desperdicio da regra fixa vs o do CRAI.
        if pol in ("regra_fixa", "crai_eprofit"):
            n = f"{desperdicio[pol]:,}".replace(",", ".")
            ax.text(12, i, f"{n} retentativas desperdicadas",
                    va="center", fontsize=8, color=SURFACE)

    ax.set_xlabel("Receita liquida (R$ mil) — faturas recuperadas menos custo das intervencoes",
                  fontsize=9, color=INK_2)
    ax.set_title("Receita liquida por politica de cobranca",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, resumo.max() / 1000 * 1.18)
    _base(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "liquido_por_politica.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_recuperacao_por_causa(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index="causa", columns="politica",
                         values="recuperou", aggfunc="mean")
    piv = piv.sort_values("regra_fixa")
    y = range(len(piv))
    h = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.barh([i + h / 2 for i in y], piv["crai_eprofit"] * 100, height=h,
            color=SERIE_1, label="CRAI (e-Profit)")
    ax.barh([i - h / 2 for i in y], piv["regra_fixa"] * 100, height=h,
            color=SERIE_2, label="Regra fixa (mercado)")

    for i, causa in enumerate(piv.index):
        ax.text(piv["crai_eprofit"][causa] * 100 + 1.5, i + h / 2,
                f"{piv['crai_eprofit'][causa]*100:.0f}%", va="center",
                fontsize=8.5, color=INK)
        ax.text(piv["regra_fixa"][causa] * 100 + 1.5, i - h / 2,
                f"{piv['regra_fixa'][causa]*100:.0f}%", va="center",
                fontsize=8.5, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels([ROTULO_CAUSA[c] for c in piv.index], fontsize=9.5, color=INK_2)
    ax.set_xlabel("Taxa de recuperacao (%)", fontsize=9, color=INK_2)
    ax.set_title("Onde a regra fixa perde: recuperacao por causa da falha",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.set_xlim(0, 112)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    _base(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recuperacao_por_causa.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_custo_vs_recuperacao(df: pd.DataFrame) -> None:
    g = df.groupby("politica").agg(custo=("custo", "sum"),
                                   recup=("recuperou", "mean"),
                                   liquido=("liquido", "sum"))

    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(g["custo"], g["recup"] * 100, s=120, color=SERIE_1,
               edgecolor=SURFACE, linewidth=2, zorder=3)

    # CRAI e oraculo quase coincidem — afastados em direcoes opostas.
    desloc = {"regra_fixa": (260, -1.0), "sempre_ligacao": (-3000, 2.2),
              "crai_eprofit": (300, -4.2), "oraculo": (-1750, 3.0)}
    for pol, row in g.iterrows():
        dx, dy = desloc.get(pol, (260, 0))
        val = f"R$ {row['liquido']/1000:,.0f}k".replace(",", ".")
        ax.annotate(f"{ROTULOS[pol]}\nliquido {val}",
                    (row["custo"] + dx, row["recup"] * 100 + dy),
                    fontsize=8.5, color=INK_2, linespacing=1.35)

    ax.set_xlabel("Custo total das intervencoes (R$)", fontsize=9, color=INK_2)
    ax.set_ylabel("Taxa de recuperacao (%)", fontsize=9, color=INK_2)
    ax.set_title("Recuperar mais nao e lucrar mais",
                 fontsize=12.5, color=INK, pad=12, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_xlim(-800, 10200)
    ax.set_ylim(44, 90)
    _base(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "custo_vs_recuperacao.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS_DIR / "simulacao.csv")
    fig_liquido(df)
    fig_recuperacao_por_causa(df)
    fig_custo_vs_recuperacao(df)
    print(f"[ok] 3 figuras salvas em {FIG_DIR}")


if __name__ == "__main__":
    main()
