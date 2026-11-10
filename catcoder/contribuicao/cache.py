import hashlib
import json
from pathlib import Path


def _nome(chave: dict) -> str:
    bruto = json.dumps(chave, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(bruto.encode("utf-8")).hexdigest() + ".json"


class Cache:
    def __init__(self, pasta: Path):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)

    def caminho(self, chave: dict) -> Path:
        return self.pasta / _nome(chave)

    def ler(self, chave: dict):
        caminho = self.caminho(chave)
        if not caminho.exists():
            return None
        return json.loads(caminho.read_text(encoding="utf-8"))

    def gravar(self, chave: dict, registro: dict):
        registro = dict(registro)
        registro["chave"] = chave
        self.caminho(chave).write_text(
            json.dumps(registro, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return registro

    def medido_brl(self) -> float:
        total = 0.0
        for caminho in self.pasta.glob("*.json"):
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            total += float(dados.get("brl") or 0)
        return total
