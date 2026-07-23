"""Testes do Modulo 5 — politica de decisao e ambiente de cobranca.

Uso:
    python -m tests.test_policy
"""
from __future__ import annotations

import sys

from src.decision_policy import DecisionPolicy
from src.environment import DunningEnvironment, MAX_ACOES
from src.policies import CraiPolicy, RegraFixaPolicy

falhas: list[str] = []


def checar(cond: bool, nome: str) -> None:
    print(f"  {'[ok]  ' if cond else '[FALHA]'} {nome}")
    if not cond:
        falhas.append(nome)


def test_ambiente() -> None:
    print("\nAmbiente")
    env = DunningEnvironment(seed=1)
    cob = env.sample_cobranca()
    checar(len(cob["_u"]) == MAX_ACOES, "sorte pre-sorteada tem um valor por passo")

    # Cartao expirado nunca passa em retentativa, em nenhum dia.
    exp = {**cob, "causa": "expired_card", "dia_liquidez": 0, "_u": [0.0] * MAX_ACOES}
    checar(not any(env.responder(exp, "retry", d, 0) for d in range(14)),
           "retry nunca resolve cartao expirado")

    # Sem saldo, nenhuma acao recupera.
    sem_saldo = {**cob, "causa": "insufficient_funds", "dia_liquidez": 9,
                 "_u": [0.0] * MAX_ACOES}
    checar(not env.responder(sem_saldo, "pix_boleto", 3, 0),
           "nenhum canal recupera antes da entrada de saldo")
    checar(env.responder(sem_saldo, "pix_boleto", 9, 0),
           "o mesmo canal recupera no dia da liquidez")

    # Comparacao pareada: mesma cobranca, mesmo passo -> mesmo resultado.
    env_a, env_b = DunningEnvironment(seed=5), DunningEnvironment(seed=99)
    c = DunningEnvironment(seed=3).sample_cobranca()
    checar(env_a.responder(c, "whatsapp", 13, 1) == env_b.responder(c, "whatsapp", 13, 1),
           "resposta independe da instancia do ambiente (comparacao pareada)")


def test_politica_decisao() -> None:
    print("\nPolitica de decisao")
    pol = DecisionPolicy()

    d = pol.decidir({"causa": "expired_card", "valor": 149.0, "p_recovery": 0.9})
    checar(d["acao"] != "retry", "cartao expirado nao recebe retentativa")

    d = pol.decidir({"causa": "insufficient_funds", "valor": 299.9,
                     "p_recovery": 0.85, "payday_previsto": 5})
    checar(d["acao"] == "retry" and d["dia_offset"] == 5,
           "saldo insuficiente agenda retry para o payday previsto")

    d = pol.decidir({"causa": "do_not_honor", "valor": 4800.0, "p_recovery": 0.35})
    checar(d["escalar_humano"], "fatura alta escala para o time de CS")

    d = pol.decidir({"causa": "do_not_honor", "valor": 4800.0, "p_recovery": 0.35},
                    cs_disponivel=False)
    checar(not d["escalar_humano"], "sem time disponivel, nao escala")

    d = pol.decidir({"causa": "do_not_honor", "valor": 0.05, "p_recovery": 0.05})
    checar(d["acao"] == "nao_intervir", "gate de e-Profit barra intervencao sem retorno")

    checar(all(pol.decidir({"causa": c, "valor": 500.0, "p_recovery": 0.7})["motivo"]
               for c in ["expired_card", "insufficient_funds", "processing_error"]),
           "toda decisao vem com raciocinio em portugues")


def test_regra_fixa() -> None:
    print("\nBaseline regra fixa")
    pol = RegraFixaPolicy()
    pol.reset(orcamento_cs=0)
    acoes = [pol.decidir({}, passo, 0) for passo in range(MAX_ACOES + 1)]
    checar(acoes[0] == (1, "retry") and acoes[1] == (3, "retry"),
           "retenta em dias fixos, ignorando o contexto")
    checar(acoes[3] == (7, "email"), "encerra com um e-mail generico")
    checar(acoes[4] is None, "para apos o orcamento de acoes")


def test_orcamento_cs() -> None:
    print("\nRacionamento do time de CS")
    pol = CraiPolicy()
    pol.reset(orcamento_cs=1)
    ctx = {"causa": "do_not_honor", "valor": 9000.0, "ltv": 50000.0,
           "p_recovery": 0.5, "payday_previsto": 0}
    primeira = pol.decidir(ctx, 0, 0)
    segunda = pol.decidir(ctx, 0, 0)
    checar(primeira[1] == "ligacao_cs", "primeira cobranca de alto valor recebe ligacao")
    checar(segunda[1] != "ligacao_cs", "orcamento esgotado impede a segunda ligacao")


if __name__ == "__main__":
    print("=" * 62)
    print("  Modulo 5 — testes da politica de decisao")
    print("=" * 62)
    test_ambiente()
    test_politica_decisao()
    test_regra_fixa()
    test_orcamento_cs()
    print("\n" + "=" * 62)
    if falhas:
        print(f"  {len(falhas)} FALHA(S): " + "; ".join(falhas))
        sys.exit(1)
    print("  todos os testes passaram")
