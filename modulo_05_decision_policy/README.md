# Modulo 5 - Politica de Decisao de Cobranca (e-Profit)

Componente do CRAI que decide **qual acao tomar** diante de uma cobranca que
falhou: retentar, contatar o cliente por algum canal, escalar para o time de
CS ou nao intervir. Substitui a regra fixa de retentativas — o alvo que o
proprio TCC critica na introducao — por uma politica que le o contexto dos
Modulos 1-3 e maximiza e-Profit esperado.

## A assimetria que a regra fixa ignora

Retentar nao resolve todas as causas:

| Causa | Retentar resolve? |
|-------|-------------------|
| Erro tecnico do gateway | Sim, quase sempre e na hora |
| Saldo insuficiente | So DEPOIS que o cliente recebe |
| Cartao expirado | **Nunca** — so o cliente pode atualizar o cartao |
| Banco nao honrou | Raramente |

A regra de mercado retenta em D+1, D+3, D+5 e manda um e-mail no fim,
independentemente da causa. Em 20% do volume (cartao expirado) as tres
retentativas sao desperdicio garantido; em 35% (saldo insuficiente) elas
acontecem antes do dinheiro entrar.

## As acoes

| Acao | Custo | Taxa de resposta |
|------|-------|------------------|
| retry | R$ 0,02 | — (depende da causa) |
| email | R$ 0,02 | 25% |
| whatsapp | R$ 0,05 | 45% |
| pix_boleto | R$ 0,50 | 55% |
| ligacao_cs | R$ 15,00 | 75% |
| nao_intervir | R$ 0,00 | — |

O time de CS e um recurso **escasso** (10% do volume). Nao basta saber que a
ligacao converte mais: e preciso decidir QUEM a recebe. A politica so escala
quando o ganho marginal sobre o melhor canal automatico passa de R$ 200.

## Pipeline

```
src/environment.py      -> ambiente simulado (verdade oculta da cobranca)
src/policies.py         -> RegraFixa, SempreLigacao, Crai, Oraculo
src/simulate.py         -> 5000 cobrancas x 4 politicas -> reports/simulacao.csv
src/evaluate.py         -> reports/metricas.json (+ ablacao do Modulo 3)
src/visualizar.py       -> reports/figures/*.png
src/decision_policy.py  -> interface para o agente (Modulo 6)
src/export_to_crai.py   -> exporta a politica calibrada para crai/models/
```

## Como rodar

```bash
pip install -r requirements.txt

python -m src.simulate
python -m src.evaluate
python -m src.visualizar
python -m src.decision_policy   # demo com raciocinio em PT-BR
python -m tests.test_policy     # testes

python -m src.export_to_crai    # exporta a politica para crai/models/
```

## Resultados (seed 42, 5.000 cobrancas)

| Politica | Receita liquida | Recuperacao | Acoes/cobranca | Retentativas desperdicadas |
|----------|----------------|-------------|----------------|----------------------------|
| Regra fixa (mercado) | R$ 1.030.619 | 48,7% | 3,11 | 7.701 |
| Sempre ligacao | R$ 1.204.673 | 57,5% | 2,34 | 0 |
| **CRAI (e-Profit)** | **R$ 1.678.437** | **79,9%** | **1,78** | **188** |
| Oraculo (teto) | R$ 1.712.591 | 81,5% | 1,70 | 104 |

Ganho vs regra fixa: **+R$ 647.818** (+62,9%, ~R$ 130 por cobranca),
alcancando **98,0% do teto do oraculo** — e com **menos** intervencoes por
cobranca (1,78 vs 3,11).

O detalhe central: `sempre_ligacao` recupera mais que a regra fixa (57,5% vs
48,7%) mas lucra R$ 474 mil a menos que o CRAI, porque queima o time de CS
sem olhar o valor da fatura. Recuperar mais nao e lucrar mais — o mesmo
argumento do Modulo 4, agora no lado involuntario.

### Contribuicao do Modulo 3

Ablacao trocando so a fonte da previsao de payday:

| Fonte do payday | Receita liquida | Recuperacao |
|-----------------|----------------|-------------|
| LSTM + Prophet (Modulo 3) | R$ 1.678.437 | 79,9% |
| Heuristica de dias fixos | R$ 1.509.048 | 72,1% |

O Modulo 3 responde por **R$ 169.389** do ganho total — 26% dele.

## Interface publica

```python
from src.decision_policy import DecisionPolicy

pol = DecisionPolicy()
pol.decidir({"causa": "expired_card", "valor": 149.0, "p_recovery": 0.90})
# {"acao": "pix_boleto", "dia_offset": 0, "eprofit_esperado": 73.26,
#  "escalar_humano": False,
#  "motivo": "Cartao expirado: Retentar nao resolve esta causa. link
#             Pix/boleto maximiza o e-Profit (R$ 73.26); a ligacao renderia
#             so R$ 12.32 a mais, abaixo do limiar."}
```

Toda decisao vem acompanhada do raciocinio em portugues — e o log de
auditoria que o agente escreve a cada no.
