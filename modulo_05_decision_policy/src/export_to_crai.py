"""Exporta a politica calibrada do Modulo 5 para o pacote crai/.

Diferente dos modulos 1-4, o artefato aqui nao sao pesos treinados: sao os
PARAMETROS da politica de decisao (crencas de retentativa por causa, taxas
de resposta por canal e o limiar de escalonamento humano), acompanhados das
metricas que os validaram na simulacao.

crai.agent.decision_policy.DecisionPolicy.load() le esse arquivo; sem ele,
cai nos mesmos valores declarados em codigo.

Uso (apos rodar src.simulate e src.evaluate):
    python -m src.export_to_crai
"""
from __future__ import annotations

import json
from pathlib import Path

from src.environment import CUSTO_ACAO
from src.policies import CRENCA_RESPOSTA, CRENCA_RETRY, LIMIAR_MARGINAL_CS

MODULO_ROOT = Path(__file__).resolve().parents[1]
CRAI_MODELS = MODULO_ROOT.parent / "crai" / "models"


def main() -> None:
    metricas_path = MODULO_ROOT / "reports" / "metricas.json"
    metricas = json.loads(metricas_path.read_text(encoding="utf-8"))

    payload = {
        "versao": "modulo_05_v1",
        "crenca_retry": CRENCA_RETRY,
        "crenca_resposta": CRENCA_RESPOSTA,
        "limiar_marginal_cs": LIMIAR_MARGINAL_CS,
        "custo_acao": CUSTO_ACAO,
        "validacao": {
            "n_cobrancas": metricas["n_cobrancas"],
            "ganho_vs_regra_fixa": metricas["ganho_crai_vs_regra_fixa"],
            "ganho_pct": metricas["ganho_pct"],
            "pct_do_teto_do_oraculo": metricas["pct_do_teto_do_oraculo"],
        },
    }

    CRAI_MODELS.mkdir(parents=True, exist_ok=True)
    destino = CRAI_MODELS / "decision_policy.json"
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] politica exportada para {destino}")


if __name__ == "__main__":
    main()
