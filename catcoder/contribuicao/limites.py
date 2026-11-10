import random

# Tetos de docs/arquitetura.md e a conta de docs/experimentos.md.
FX_BRL = 5.57
TETO_BRL = 20.0
MAX_VOLTAS = 5
MAX_TENTATIVAS = 5
MAX_GRAU = 4
MAX_TRECHOS = 16
MAX_CHARS_TRECHO = 1_000
MAX_CHARS_TIPO = 12_000
MAX_CHARS_DOC_SIG = 2_000
MAX_CHARS_VERSAO = 6_000
MAX_CHARS_VERSAO_JUIZA = 1_500
MAX_OUT_N = 32
MAX_OUT_GERADORA = 1024
MAX_OUT_JUIZA = 256

MODELO_FRACO = "gpt-4o-mini"
MODELO_FORTE = "gpt-5"
MODELO_N = MODELO_FRACO

# USD por 1 milhão de tokens, tabela da OpenAI em 10 nov 2026.
PRECO = {
    MODELO_FRACO: (0.15, 0.60),
    MODELO_FORTE: (1.25, 10.00),
}

# Reserva de uma tarefa no teto de 5 voltas (entrada/saída da seção Tokens).
TOKENS_N = (500, 32)
TOKENS_GERADORA = (44_500, 5_120)
TOKENS_JUIZA = (47_875, 1_280)


def custo_usd(modelo, entrada, saida):
    pin, pout = PRECO[modelo]
    return entrada / 1e6 * pin + saida / 1e6 * pout


def reserva_usd(geradora, juiza):
    return (
        custo_usd(MODELO_N, *TOKENS_N)
        + custo_usd(geradora, *TOKENS_GERADORA)
        + custo_usd(juiza, *TOKENS_JUIZA)
    )


def reserva_brl(geradora, juiza):
    return reserva_usd(geradora, juiza) * FX_BRL


def _embaralhar(n):
    rng = random.Random(0)
    xs = list(range(n))
    rng.shuffle(xs)
    return xs


RUST_N = 90
JAVA_N = 199
RUST_IDX = _embaralhar(RUST_N)
JAVA_IDX = _embaralhar(JAVA_N)

RODADAS = [
    {"id": "R-FF", "lingua": "rust", "geradora": MODELO_FRACO, "juiza": MODELO_FRACO, "indices": RUST_IDX[:46]},
    {"id": "R-JF", "lingua": "rust", "geradora": MODELO_FRACO, "juiza": MODELO_FORTE, "indices": RUST_IDX[:8]},
    {"id": "R-FFORTE", "lingua": "rust", "geradora": MODELO_FORTE, "juiza": MODELO_FORTE, "indices": RUST_IDX[:4]},
    {"id": "J-FF", "lingua": "java", "geradora": MODELO_FRACO, "juiza": MODELO_FRACO, "indices": JAVA_IDX[:30]},
    {"id": "J-JF", "lingua": "java", "geradora": MODELO_FRACO, "juiza": MODELO_FORTE, "indices": JAVA_IDX[:6]},
    {"id": "J-FFORTE", "lingua": "java", "geradora": MODELO_FORTE, "juiza": MODELO_FORTE, "indices": JAVA_IDX[:2]},
]
