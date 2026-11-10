import tempfile
from pathlib import Path

from .cache import Cache
from .ciclo import cortar, trajetoria, trechos_rag
from .execucao import pode_comecar
from .figuras import _amostras_doc
from .limites import MAX_CHARS_TRECHO, MAX_TRECHOS, TETO_BRL


class Falso:
    def __init__(self):
        self.chamadas = 0

    def completar(self, *, modelo, system, user, max_output_tokens):
        self.chamadas += 1
        if max_output_tokens == 32:
            texto = "3"
        elif max_output_tokens == 256:
            texto = '{"acao": "parar", "versao": 0, "feedback": "ok"}'
        else:
            texto = "fn f() { return 1; }"
        return {
            "texto": texto,
            "input_tokens": 10,
            "output_tokens": 4,
            "usd": 0.001,
            "brl": 0.005,
            "modelo": modelo,
        }


def _checar_corte():
    rag = "\n\n".join(f"trecho {i} " + ("x" * 2000) for i in range(20))
    saida = trechos_rag(rag)
    partes = saida.split("\n\n")
    assert len(partes) == MAX_TRECHOS
    assert all(len(p) <= MAX_CHARS_TRECHO for p in partes)
    assert len(cortar("abc", 2)) == 2


def _checar_cache():
    with tempfile.TemporaryDirectory() as pasta:
        cache = Cache(Path(pasta))
        cliente = Falso()

        def montar(grau):
            return f"prompt grau {grau}"

        def testar(codigo):
            assert "fn f" in codigo
            return True, True

        trajetoria(
            cliente, cache,
            lingua="rust", indice=8, rodada="R-FF",
            geradora="gpt-4o-mini", juiza="gpt-4o-mini",
            doc="doc", assinatura="fn f()",
            montar_prompt=montar, testar=testar,
        )
        feitas = cliente.chamadas
        assert feitas == 3
        trajetoria(
            cliente, cache,
            lingua="rust", indice=8, rodada="R-FF",
            geradora="gpt-4o-mini", juiza="gpt-4o-mini",
            doc="doc", assinatura="fn f()",
            montar_prompt=montar, testar=testar,
        )
        assert cliente.chamadas == feitas
        trajetoria(
            cliente, cache,
            lingua="rust", indice=8, rodada="R-JF",
            geradora="gpt-4o-mini", juiza="gpt-5",
            doc="doc", assinatura="fn f()",
            montar_prompt=montar, testar=testar,
        )
        assert cliente.chamadas == feitas + 2


def _checar_teto():
    assert pode_comecar(19.0, 0.09)
    assert not pode_comecar(19.9, 0.46)
    assert TETO_BRL == 20.0


def main():
    _amostras_doc()
    _checar_corte()
    _checar_cache()
    _checar_teto()
    print("selfcheck ok")


if __name__ == "__main__":
    main()
