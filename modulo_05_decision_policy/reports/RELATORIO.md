# Relatorio - Modulo 5: Politica de Decisao de Cobranca

**Projeto:** CRAI - Churn Recovery Artificial Intelligence (TCC)
**Modulo:** 5 de 6 - Decision Policy (orientada a e-Profit)
**Data:** julho/2026
**Seed:** 42 (reproduzivel)

---

## 1. Objetivo

Os Modulos 1-4 produzem informacao: causa raiz e recuperabilidade (1),
anomalia comportamental (2), janela de liquidez (3) e oferta otima de
retencao (4). Falta a camada que **age** sobre ela no churn involuntario.

O Modulo 5 e a politica de decisao: dada uma cobranca que falhou, escolhe a
proxima acao — retentar, contatar o cliente por e-mail, WhatsApp ou
Pix/boleto, escalar para o time de CS, ou nao intervir — maximizando o
e-Profit esperado. E, para cada decisao, registra o raciocinio em portugues.

## 2. O baseline: a regra fixa que o CRAI existe para substituir

A introducao deste TCC afirma que "a maioria usa regras fixas de retentativa
que ignoram o contexto do cliente". O Modulo 5 e onde essa afirmacao vira
numero. O baseline implementado e exatamente essa regra: retentativas em
D+1, D+3 e D+5, seguidas de um e-mail generico em D+7, sem olhar causa,
valor, LTV nem previsao de liquidez.

O que a regra fixa ignora e mensuravel:

| Causa | Share | O retry resolve? |
|-------|------:|------------------|
| insufficient_funds | 35% | So depois da entrada de saldo |
| expired_card | 20% | **Nunca** |
| card_declined | 15% | Raramente (15%) |
| do_not_honor | 12% | Quase nunca (5%) |
| processing_error | 10% | Quase sempre (90%), na hora |
| generic_decline | 8% | As vezes (25%) |

Em um quinto do volume as tres retentativas sao desperdicio garantido — o
cartao esta vencido e nenhuma tentativa automatica vai passar. Em outro
terco, elas acontecem antes de o cliente ter dinheiro.

## 3. O ambiente simulado

Verdade oculta por cobranca: causa, perfil de pagamento (CLT/PJ/freelancer),
valor da fatura, LTV, e o **dia em que o cliente volta a ter saldo**. Duas
regras governam o resultado de qualquer acao:

- retentar sem saldo falha com certeza;
- pedir Pix/boleto sem saldo tambem falha — nao adianta um canal melhor se
  nao ha dinheiro do outro lado.

A sorte de cada cobranca e **pre-sorteada** (um valor por passo), entao todas
as politicas enfrentam exatamente o mesmo acaso mesmo tomando numeros
diferentes de acoes. E uma comparacao pareada, nao uma media de execucoes
independentes.

O time de CS e um recurso escasso: 10% do volume (500 ligacoes para 5.000
cobrancas). Esse limite e o que torna a decisao interessante — sem ele, a
resposta trivial seria "ligue para todo mundo".

## 4. As politicas comparadas

| Politica | O que usa |
|----------|-----------|
| `regra_fixa` | nada — dias fixos |
| `sempre_ligacao` | liga enquanto houver time, depois WhatsApp |
| `crai_eprofit` | causa (M1), recuperabilidade (M1), valor, payday (M3) |
| `oraculo` | a verdade oculta do ambiente (teto pratico) |

A politica CRAI escolhe, a cada passo, a acao de maior e-Profit esperado:

```
e-Profit(acao) = P(recuperar | acao) x valor da fatura - custo(acao)
```

com `P(recuperar | retry)` vindo da crenca por causa e
`P(recuperar | canal) = p_recovery x taxa_de_resposta(canal)`. O
escalonamento humano exige que o **ganho marginal** sobre o melhor canal
automatico supere R$ 200 — e assim que o recurso escasso e racionado.

## 5. Resultados

| Politica | Receita | Custo | **Liquido** | Recuperacao | Acoes/cobranca | Retries desperdicados |
|----------|--------:|------:|------------:|------------:|---------------:|----------------------:|
| Regra fixa | R$ 1.030.930 | R$ 311 | R$ 1.030.619 | 48,7% | 3,11 | 7.701 |
| Sempre ligacao | R$ 1.212.731 | R$ 8.059 | R$ 1.204.673 | 57,5% | 2,34 | 0 |
| **CRAI (e-Profit)** | R$ 1.683.074 | R$ 4.637 | **R$ 1.678.437** | **79,9%** | 1,78 | 188 |
| Oraculo (teto) | R$ 1.717.009 | R$ 4.418 | R$ 1.712.591 | 81,5% | 1,70 | 104 |

- Ganho vs regra fixa: **+R$ 647.818** (+62,9%), ~R$ 130 por cobranca.
- **98,0% do teto do oraculo** — a politica captura quase todo o valor que
  seria acessivel conhecendo a verdade oculta.
- Faz mais com **menos**: 1,78 acoes por cobranca contra 3,11 da regra fixa,
  e 41x menos retentativas desperdicadas (188 vs 7.701).

Figuras: `figures/liquido_por_politica.png`,
`figures/recuperacao_por_causa.png`, `figures/custo_vs_recuperacao.png`.

### 5.1 Recuperar mais nao e lucrar mais

`sempre_ligacao` recupera 57,5% contra 48,7% da regra fixa, mas gasta 26x
mais em intervencoes e fica R$ 474 mil atras do CRAI. Ela queima as 500
ligacoes disponiveis nas primeiras cobrancas que aparecem, sem olhar o valor
da fatura. O CRAI usa apenas **54** ligacoes — as que passam do limiar
marginal — e lucra mais. E o mesmo argumento do Modulo 4, agora no lado
involuntario: otimizar conversao e otimizar lucro sao objetivos diferentes.

### 5.2 Onde a regra fixa perde

Taxa de recuperacao por causa:

| Causa | Regra fixa | CRAI | Delta |
|-------|-----------:|-----:|------:|
| Cartao expirado | 22,4% | 87,9% | **+65,5 pp** |
| Saldo insuficiente | 56,1% | 90,5% | +34,4 pp |
| Banco nao honrou | 21,6% | 44,7% | +23,1 pp |
| Cartao recusado | 45,7% | 63,2% | +17,5 pp |
| Recusa generica | 65,4% | 69,8% | +4,4 pp |
| Erro tecnico | 100,0% | 97,6% | **-2,4 pp** |

O ganho se concentra exatamente onde a teoria previa: cartao expirado, em
que a regra fixa gasta tres retentativas garantidamente inuteis antes de
tentar falar com o cliente.

**A regra fixa vence em erro tecnico** (100% vs 97,6%). E um resultado
honesto e explicavel: falha transitoria de gateway se resolve com
insistencia barata, e retentar tres vezes cobre praticamente todos os casos,
enquanto o CRAI encerra assim que o e-Profit marginal deixa de compensar.
Insistencia cega e a estrategia certa em 10% do volume — e errada nos
outros 90%.

### 5.3 Quanto do ganho vem do Modulo 3

Ablacao trocando apenas a fonte da previsao de payday, mantendo o resto
identico:

| Fonte do payday | Liquido | Recuperacao |
|-----------------|--------:|------------:|
| LSTM + Prophet (Modulo 3) | R$ 1.678.437 | 79,9% |
| Heuristica de dias fixos | R$ 1.509.048 | 72,1% |

O Modulo 3 responde por **R$ 169.389** — 26% do ganho total do Modulo 5.
Esse numero so existe porque a politica agenda a retentativa para o dia
previsto: um erro de 5 dias na previsao coloca a tentativa antes do salario
cair, e ela falha com certeza.

## 6. Integracao ao pipeline (crai/)

- `crai/agent/decision_policy.py`: a politica em runtime, carregando os
  parametros calibrados de `crai/models/decision_policy.json` (sem o
  arquivo, cai nos mesmos priors declarados em codigo).
- `crai/agent/workflow.py`: novo no `decide_action`, entre `infer_payday` e
  `schedule_retry`.
- `crai/agent/main_agent.py`: roteamento condicional de 3 vias —
  `retry` -> `schedule_retry`; canal -> `trigger_dunning`; `nao_intervir` ->
  `update_dashboard`.
- `crai/dunning/dunning_engine.py`: passa a receber o canal decidido em vez
  de assumir WhatsApp.
- Cada decisao loga o raciocinio em PT-BR, por exemplo:

  > `[RACIOCINIO] Cartao expirado: Retentar nao resolve esta causa. link
  > Pix/boleto maximiza o e-Profit (R$ 68.42); a ligacao renderia so
  > R$ 10.56 a mais, abaixo do limiar.`

## 7. Limitacoes conhecidas

- **As taxas de resposta por canal sao priors declarados**, calibrados em
  benchmarks, nao aprendidos. Sao a mesma classe de suposicao que o
  `SEED_PRIORS` do Modulo 4 — mas ali os posteriores sao atualizados
  online, e aqui ainda nao.
- **O limiar de R$ 200 para escalonamento humano e um parametro fixo.** Ele
  raciona o time de CS de forma aproximadamente correta neste volume, mas
  nao se adapta se a capacidade mudar. O certo seria um preco-sombra do
  recurso, ajustado pela ocupacao da fila.
- **O ganho e contabilizado sobre o valor da fatura**, nao sobre o LTV. E
  conservador de proposito: nao credita a politica pela retencao do cliente,
  so pela cobranca recuperada.
- **O ambiente e estacionario e sem efeito de fadiga**: contatar o mesmo
  cliente tres vezes tem o mesmo efeito da primeira. Na pratica ha desgaste,
  e ele penalizaria as politicas mais insistentes — ou seja, a regra fixa
  apareceria ainda pior.
- **A causa e assumida como observavel e correta** (vem do codigo do
  gateway). Erros de diagnostico do Modulo 1 nao se propagam aqui.

## 8. Proximos passos

- Aprender as taxas de resposta por canal online, com a mesma estrutura
  Beta-Bernoulli do Modulo 4 — trocando o par (perfil, oferta) por
  (causa, canal). Unificaria os dois lados do CRAI sob o mesmo mecanismo.
- Preco-sombra do time de CS em vez de limiar fixo.
- Modelar fadiga de contato e o custo reputacional da insistencia.
- Levar a mesma correcao ao Modulo 1: hoje `_find_optimal_channel` maximiza
  `p x LTV - custo` sem taxa de resposta por canal, o que faz o canal otimo
  degenerar sempre para o mais barato.

## 9. Artefatos

| Arquivo | Conteudo |
|---------|----------|
| `models/meta.json` | parametros da simulacao |
| `reports/metricas.json` | todas as metricas desta pagina |
| `reports/simulacao.csv` | uma linha por (politica, cobranca) |
| `reports/figures/*.png` | figuras para a banca |
| `crai/models/decision_policy.json` | politica calibrada exportada |
