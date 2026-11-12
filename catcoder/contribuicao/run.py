import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

from .cache import Cache
from .ciclo import OpenAICliente
from .execucao import RAIZ, SAIDA, executar
from .figuras import gerar
from .limites import TETO_BRL


def _cargo_no_path():
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        import os
        os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ.get("PATH", "")


def preflight():
    _cargo_no_path()
    rust = bool(shutil.which("cargo") and shutil.which("rust-analyzer"))
    java = bool(shutil.which("defects4j"))
    return rust, java


def main():
    load_dotenv(RAIZ / ".env")
    rust_ok, java_ok = preflight()
    print(f"preflight rust={rust_ok} java={java_ok}")
    if not rust_ok and not java_ok:
        print("Nenhuma língua pode rodar. Nenhuma chamada de API foi feita.")
        gerar(SAIDA, [])
        return
    cache = Cache(SAIDA / "cache")
    cliente = OpenAICliente()
    motivos = executar(cliente, cache, rust_ok, java_ok)
    ledger = {
        "medido_brl": cache.medido_brl(),
        "teto_brl": TETO_BRL,
        "motivos": motivos,
        "rust": rust_ok or any((SAIDA / "tarefas").glob("R-*.json")),
        "java": java_ok or any((SAIDA / "tarefas").glob("J-*.json")),
    }
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "ledger.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    gerar(SAIDA, motivos)
    print(f"medido R$ {ledger['medido_brl']:.4f}")


if __name__ == "__main__":
    main()
