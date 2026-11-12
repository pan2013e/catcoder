import importlib.util
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .cache import Cache
from .ciclo import (
    cortar,
    doc_e_assinatura,
    trajetoria,
    trechos_rag,
)
from .limites import (
    MAX_CHARS_TIPO,
    MAX_GRAU,
    RODADAS,
    TETO_BRL,
    reserva_brl,
)

RAIZ = Path(__file__).resolve().parents[2]
RUST_DIR = RAIZ / "catcoder" / "rust"
JAVA_DIR = RAIZ / "catcoder" / "java"
INTEL = RAIZ / "catcoder" / "tools" / "intellirust"
SAIDA = RAIZ / "results" / "contribuicao"
TAREFAS = SAIDA / "tarefas"
TIPO = SAIDA / "cache_tipo"


def _modulo(nome, caminho):
    spec = importlib.util.spec_from_file_location(nome, caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linha(item):
    if hasattr(item, "keys"):
        return {k: item[k] for k in item.keys()}
    return dict(item)


def _assinatura(lingua, item):
    if lingua == "rust":
        return item.get("signature") or ""
    return item.get("focal_fn_signature") or ""


def pode_comecar(medido, reserva, teto=TETO_BRL):
    return medido + reserva <= teto


def extrair_tipo_rust(item, indice, grau):
    TIPO.mkdir(parents=True, exist_ok=True)
    caminho = TIPO / f"rust_{indice}_n{grau}.txt"
    if caminho.exists():
        return caminho.read_text(encoding="utf-8")
    if str(INTEL) not in sys.path:
        sys.path.insert(0, str(INTEL))
    if str(RUST_DIR) not in sys.path:
        sys.path.insert(0, str(RUST_DIR))
    extrator = _modulo("extract_type_context_contrib", RUST_DIR / "extract_type_context.py")
    bruto = extrator.make_ctx(
        (item, indice),
        degree=grau,
        crates_base=str(RUST_DIR / "crates"),
    )["extended_context"]
    texto = cortar(bruto, MAX_CHARS_TIPO)
    caminho.write_text(texto, encoding="utf-8")
    return texto


def tipo_java(item):
    return cortar(item.get("extended_context"), MAX_CHARS_TIPO)


def _prompt(lingua, item, tipo):
    doc, sig = doc_e_assinatura(item.get("docstring"), _assinatura(lingua, item))
    dados = {
        "focal_fn_signature": sig,
        "docstring": doc,
        "focal_ctx": tipo,
        "rag_data": trechos_rag(item.get("rag_data")),
    }
    pasta = RUST_DIR if lingua == "rust" else JAVA_DIR
    if str(pasta) not in sys.path:
        sys.path.insert(0, str(pasta))
    nome = "util_rust_contrib" if lingua == "rust" else "util_java_contrib"
    util = _modulo(nome, pasta / "util.py")
    return util.build_prompt(dados, True)


def testar_rust(item, codigo):
    if str(RUST_DIR) not in sys.path:
        sys.path.insert(0, str(RUST_DIR))
    adapt = _modulo("test_adapter_rust_contrib", RUST_DIR / "test_adapter.py")
    dados = dict(item)
    dados["focal_fn_full"] = codigo
    with adapt.TestAdapter(str(RUST_DIR / "crates"), dados, False) as caixa:
        ok = caixa.test()
        return bool(caixa.compile_success), bool(ok)


def testar_java(item, codigo):
    if str(JAVA_DIR) not in sys.path:
        sys.path.insert(0, str(JAVA_DIR))
    adapt = _modulo("test_adapter_java_contrib", JAVA_DIR / "test_adapter.py")
    dados = dict(item)
    dados["focal_fn_full"] = codigo
    with adapt.TestAdapter(dados) as caixa:
        ok = caixa.test()
        return bool(caixa.compile_success), bool(ok)


def _dataset(lingua):
    from datasets import load_from_disk
    pasta = RUST_DIR / "dataset" / "rusteval" if lingua == "rust" else JAVA_DIR / "dataset" / "javaeval"
    return load_from_disk(str(pasta))


def executar(cliente, cache, rust_ok, java_ok):
    TAREFAS.mkdir(parents=True, exist_ok=True)
    bases = {}
    motivos = []
    pulou = set()
    for rodada in RODADAS:
        lingua = rodada["lingua"]
        pendentes = []
        for indice in rodada["indices"]:
            destino = TAREFAS / f"{rodada['id']}_{indice}.json"
            if destino.exists():
                anterior = json.loads(destino.read_text(encoding="utf-8"))
                if anterior.get("completo"):
                    continue
            pendentes.append(indice)
        if not pendentes:
            continue
        if lingua == "rust" and not rust_ok:
            if "rust" not in pulou:
                motivos.append("Rust não rodou: cargo ou rust-analyzer ausente.")
                pulou.add("rust")
            continue
        if lingua == "java" and not java_ok:
            if "java" not in pulou:
                motivos.append("Java não rodou: defects4j não respondeu.")
                pulou.add("java")
            continue
        if lingua not in bases:
            bases[lingua] = _dataset(lingua)
        base = bases[lingua]
        reserva = reserva_brl(rodada["geradora"], rodada["juiza"])
        for indice in pendentes:
            destino = TAREFAS / f"{rodada['id']}_{indice}.json"
            if not pode_comecar(cache.medido_brl(), reserva):
                motivos.append(
                    f"Parou antes de {rodada['id']} índice {indice}: teto de R$ {TETO_BRL:.2f}."
                )
                return motivos
            item = _linha(base[indice])
            sig = _assinatura(lingua, item)
            doc = item.get("docstring") or ""

            def montar(grau, item=item, lingua=lingua, indice=indice):
                if lingua == "rust":
                    tipo = extrair_tipo_rust(item, indice, grau)
                else:
                    tipo = tipo_java(item)
                return _prompt(lingua, item, tipo)

            def testar(codigo, item=item, lingua=lingua):
                if lingua == "rust":
                    return testar_rust(item, codigo)
                return testar_java(item, codigo)

            try:
                resultado = trajetoria(
                    cliente, cache,
                    lingua=lingua,
                    indice=indice,
                    rodada=rodada["id"],
                    geradora=rodada["geradora"],
                    juiza=rodada["juiza"],
                    doc=doc,
                    assinatura=sig,
                    montar_prompt=montar,
                    testar=testar,
                )
                resultado["erro"] = ""
            except RuntimeError as exc:
                destino.write_text(json.dumps({
                    "completo": True, "erro": str(exc), "passou": False, "compilou": False,
                    "indice": indice, "lingua": lingua, "rodada": rodada["id"],
                    "geradora": rodada["geradora"], "juiza": rodada["juiza"],
                    "n_corte": MAX_GRAU, "medido_brl": cache.medido_brl(),
                    "n_pedido": None, "n_usado": None, "voltas": 0, "acoes": [],
                    "acao_final": "", "codigo": "", "tokens_in": 0, "tokens_out": 0, "brl": 0.0,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                motivos.append(str(exc))
                return motivos
            except Exception as exc:
                resultado = {
                    "n_pedido": None, "n_usado": None, "voltas": 0, "acoes": [],
                    "acao_final": "", "codigo": "", "compilou": False, "passou": False,
                    "tokens_in": 0, "tokens_out": 0, "brl": 0.0, "erro": str(exc),
                }
            resultado.update({
                "completo": True,
                "indice": indice,
                "lingua": lingua,
                "rodada": rodada["id"],
                "geradora": rodada["geradora"],
                "juiza": rodada["juiza"],
                "n_corte": MAX_GRAU,
                "medido_brl": cache.medido_brl(),
            })
            destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"{rodada['id']} #{indice} compilou={resultado['compilou']} passou={resultado['passou']} "
                f"voltas={resultado['voltas']} R$={resultado['medido_brl']:.4f} {resultado.get('erro', '')}",
                flush=True,
            )
    return motivos
