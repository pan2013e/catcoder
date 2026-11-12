import json
from pathlib import Path

from .limites import (
    JAVA_IDX,
    RODADAS,
    RUST_IDX,
    TETO_BRL,
    reserva_brl,
)


MODELOS_JAVA = [
    ("CG-7B", "codegemma-7b"),
    ("ML-8B", "llama3-8b"),
    ("CL-7B", "codellama-7b"),
    ("CQ-7B", "codeqwen-7b"),
    ("DS-6.7B", "deepseek-coder-6.7b"),
    ("CL-13B", "codellama-13b"),
]


def _pass1(pasta, trecho):
    pasta = Path(pasta)
    if not pasta.is_dir():
        return None
    for caminho in pasta.glob("*.json"):
        if trecho in caminho.name:
            dados = json.loads(caminho.read_text(encoding="utf-8"))
            return dados.get("lastest_eval", {}).get("pass@1")
    return None


ORDEM = {
    "java": ["CG-7B", "ML-8B", "Média", "CL-7B", "CQ-7B", "DS-6.7B", "CL-13B"],
    "rust": ["ML-8B", "CG-7B", "CL-7B", "Média", "DS-6.7B", "CQ-7B", "CL-13B"],
}
# Cores medidas na legenda dos slides do artigo.
CORES = {
    "In-File": "#68B098",
    "RepoCoder": "#D88870",
    "Our Approach": "#8898B8",
    "FULL": "#68B098",
    "-CR": "#D88870",
    "-TC": "#8898B8",
}
PARES_JAVA = [
    ("J-FF", "gpt-6-luna\njuíza gpt-6-luna", "#68B098"),
    ("J-JF", "gpt-6-luna\njuíza gpt-5", "#D88870"),
    ("J-FFORTE", "gpt-5\njuíza gpt-5", "#8898B8"),
]
PARES_RUST = [
    ("R-FF", "4o-mini\njuíza 4o-mini", "#68B098"),
    ("R-JF", "4o-mini\njuíza gpt-5", "#D88870"),
    ("R-FFORTE", "gpt-5\njuíza gpt-5", "#8898B8"),
]
PARES = PARES_JAVA + PARES_RUST


def _pontos(raiz, lingua, metodo):
    pasta = raiz / "results" / lingua / metodo
    pontos = {rotulo: _pass1(pasta, trecho) for rotulo, trecho in MODELOS_JAVA}
    valores = [pontos[rotulo] for rotulo, _ in MODELOS_JAVA]
    pontos["Média"] = sum(valores) / len(valores) if all(v is not None for v in valores) else None
    return pontos


def _rotulos(lingua, series):
    return [nome for nome in ORDEM[lingua] if any(serie.get(nome) is not None for serie in series)]


def _ler_tarefas(saida):
    pasta = Path(saida) / "tarefas"
    if not pasta.is_dir():
        return []
    linhas = []
    for caminho in pasta.glob("*.json"):
        linhas.append(json.loads(caminho.read_text(encoding="utf-8")))
    return linhas


def resumo(saida):
    linhas = _ler_tarefas(saida)
    rodadas = []
    for rodada in RODADAS:
        delas = [ln for ln in linhas if ln.get("rodada") == rodada["id"] and ln.get("completo")]
        n = len(delas)
        sucessos = sum(1 for ln in delas if ln.get("passou"))
        compilou = sum(1 for ln in delas if ln.get("compilou"))
        acoes = {"continua": 0, "reverte": 0, "parar": 0}
        voltas = []
        graus = []
        for ln in delas:
            voltas.append(ln.get("voltas") or 0)
            if ln.get("n_usado") is not None:
                graus.append(ln["n_usado"])
            for acao in ln.get("acoes") or []:
                if acao in acoes:
                    acoes[acao] += 1
        rodadas.append({
            "id": rodada["id"],
            "lingua": rodada["lingua"],
            "geradora": rodada["geradora"],
            "juiza": rodada["juiza"],
            "n_planejado": len(rodada["indices"]),
            "n_completas": n,
            "taxa_sucesso": (sucessos / n) if n else None,
            "taxa_compilacao": (compilou / n) if n else None,
            "media_voltas": (sum(voltas) / len(voltas)) if voltas else None,
            "media_n": (sum(graus) / len(graus)) if graus else None,
            "acoes": acoes,
            "brl": sum(float(ln.get("brl") or 0) for ln in delas),
            "tokens_in": sum(int(ln.get("tokens_in") or 0) for ln in delas),
            "tokens_out": sum(int(ln.get("tokens_out") or 0) for ln in delas),
            "reserva_brl": reserva_brl(rodada["geradora"], rodada["juiza"]),
        })
    return {"teto_brl": TETO_BRL, "rodadas": rodadas}


def _estilo(plt):
    plt.rcParams.update({
        "font.family": "Segoe UI",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#1C1C1C",
        "axes.labelcolor": "#1C1C1C",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def _eixo_artigo(ax, rotulos, series):
    import numpy as np
    from matplotlib.patches import Patch

    x = np.arange(len(rotulos))
    n = max(1, len(series))
    largura = 0.76 / n
    for j in range(len(rotulos)):
        presentes = [(nome, valores[j]) for nome, valores in series if valores[j] is not None]
        m = len(presentes)
        for i, (nome, valor) in enumerate(presentes):
            desloc = (i - (m - 1) / 2) * largura
            ax.bar(x[j] + desloc, valor, largura * 0.9, color=CORES[nome], zorder=3, linewidth=0)
    ax.set_xlim(-0.55, max(0, len(rotulos) - 0.45))
    ax.set_ylim(0, 0.62)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_ylabel("pass@1", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, fontsize=10)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#C8C8C8")
    legenda = ax.legend(
        handles=[Patch(facecolor=CORES[nome], edgecolor="none", label=nome) for nome, _ in series],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=True,
        fontsize=9,
        edgecolor="#E4E4E4",
        borderpad=0.6,
        handlelength=1.1,
        handleheight=0.7,
        columnspacing=1.2,
    )
    legenda.get_frame().set_linewidth(0.6)
    return legenda


def _painel_artigo(ax, raiz, lingua, metodos, letra, titulo):
    series = [(nome, _pontos(raiz, lingua, pasta)) for nome, pasta in metodos]
    rotulos = _rotulos(lingua, [pontos for _, pontos in series])
    _eixo_artigo(ax, rotulos, [(nome, [pontos.get(r) for r in rotulos]) for nome, pontos in series])
    ax.set_xlabel(f"({letra})  {titulo}", fontsize=13, labelpad=8)


def _painel_contrib(ax, por_id, letra, titulo, pares=PARES_RUST):
    import numpy as np

    rotulos, valores, cores, textos = [], [], [], []
    for rid, rotulo, cor in pares:
        item = por_id.get(rid)
        if not item or not item["n_completas"] or item["taxa_sucesso"] is None:
            continue
        sucessos = int(round(item["taxa_sucesso"] * item["n_completas"]))
        rotulos.append(rotulo)
        valores.append(item["taxa_sucesso"])
        cores.append(cor)
        textos.append(f"{sucessos} de {item['n_completas']}")
    x = np.arange(len(rotulos))
    barras = ax.bar(x, valores, 0.52, color=cores, zorder=3, linewidth=0)
    for barra, valor in zip(barras, valores):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + 0.025,
            f"{valor * 100:.1f}%".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#1C1C1C",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{rotulo}\n{texto}" for rotulo, texto in zip(rotulos, textos)], fontsize=10)
    ax.set_xlim(-0.6, max(0, len(rotulos) - 0.4))
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("taxa de sucesso", fontsize=11)
    ax.yaxis.grid(True, color="#E4E4E4", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#C8C8C8")
    ax.set_xlabel(f"({letra})  {titulo}", fontsize=13, labelpad=8)


def _figura(plt, titulo, subtitulo, nota):
    fig = plt.figure(figsize=(13.6, 8.8))
    grade = fig.add_gridspec(
        2, 2, height_ratios=[1.15, 1.0],
        left=0.06, right=0.985, top=0.84, bottom=0.16,
        hspace=0.62, wspace=0.26,
    )
    fig.text(0.012, 0.975, titulo, fontsize=22, fontweight="bold", color="#1A1A1A", va="top")
    fig.text(0.012, 0.925, subtitulo, fontsize=11, color="#444444", va="top")
    fig.text(0.012, 0.012, nota, fontsize=8.5, color="#666666", va="bottom", wrap=True)
    return fig, grade


def _pct(valor):
    if valor is None:
        return "—"
    return f"{valor * 100:.1f}%".replace(".", ",")


def _tabela_artigo(raiz, lingua, metodos):
    series = [(nome, _pontos(raiz, lingua, pasta)) for nome, pasta in metodos]
    rotulos = _rotulos(lingua, [pontos for _, pontos in series])
    linhas = ["| Modelo | " + " | ".join(nome for nome, _ in series) + " |",
              "|---|" + "---|" * len(series)]
    for rotulo in rotulos:
        celulas = " | ".join(_pct(pontos.get(rotulo)) for _, pontos in series)
        linhas.append(f"| {rotulo} | {celulas} |")
    return "\n".join(linhas)


def _descrever(saida, raiz, dados, linhas):
    del linhas
    por_id = {item["id"]: item for item in dados["rodadas"]}
    blocos = []
    for rid, rotulo, _ in PARES:
        item = por_id[rid]
        lingua_nome = "Java" if item["lingua"] == "java" else "Rust"
        sucessos = int(round((item["taxa_sucesso"] or 0) * item["n_completas"]))
        compilou = int(round((item["taxa_compilacao"] or 0) * item["n_completas"]))
        voltas = "—" if item["media_voltas"] is None else f"{item['media_voltas']:.1f}".replace(".", ",")
        grau = "—" if item["media_n"] is None else f"{item['media_n']:.1f}".replace(".", ",")
        reais = f"{item['brl']:.2f}".replace(".", ",")
        blocos.append(
            f"| {item['id']} ({lingua_nome}) | {rotulo.replace(chr(10), ' / ')} | {sucessos} de {item['n_completas']} | "
            f"{_pct(item['taxa_sucesso'])} | {compilou} de {item['n_completas']} ({_pct(item['taxa_compilacao'])}) | "
            f"{voltas} | {grau} | R$ {reais} |"
        )
    general = [("In-File", "infile"), ("RepoCoder", "repocoder"), ("Our Approach", "catcoder")]
    ablacao = [("FULL", "catcoder"), ("-CR", "-cr"), ("-TC", "-tc")]
    soma_brl = f"{sum(item['brl'] for item in dados['rodadas']):.2f}".replace(".", ",")
    medido = f"{float(dados.get('medido_brl') or 0):.2f}".replace(".", ",")
    texto = f"""# Resultados para os slides

Os painéis do artigo repetem o recorte dos slides: In-File, RepoCoder e Our Approach na generalização; FULL, -CR e -TC na ablação. As cores são as da legenda daqueles slides. A ordem dos modelos em Java é CG-7B, ML-8B, Média, CL-7B, CQ-7B, DS-6.7B, CL-13B. Em Rust, a ordem do slide seria ML-8B, CG-7B, CL-7B, Média, DS-6.7B, CQ-7B, CL-13B.

Cada número do artigo é o `pass@1` de `lastest_eval` no JSON de `results/`. A Média só entra quando os seis modelos existem. Modelo sem JSON fica de fora da figura e da tabela, em branco. Nada foi lido a partir do screenshot.

A contribuição fica no painel de baixo, com outro eixo. A taxa é a fração de tarefas da amostra em que o código entregue compilou e passou em todos os testes. É uma trajetória por tarefa. Esse número não é o pass@1 do artigo e não entra na Média.

## Generalização

Arquivo: `figuras/generalizacao.png`.

### Java, pass@1 do artigo

{_tabela_artigo(raiz, "java", general)}

O ganho de Our Approach sobre In-File fica entre 13 e 16 pontos em CG-7B, ML-8B e DS-6.7B, e perto de 9 pontos em CL-7B e CL-13B. Sobre RepoCoder, o ganho fica entre 3 e 8 pontos.

### Rust, pass@1 do artigo

{_tabela_artigo(raiz, "rust", general)}

Neste repositório o Rust do artigo só tem CodeLlama-13B. Os outros modelos do slide ficam em branco.

## Ablação

Arquivo: `figuras/ablacao.png`.

### Java, pass@1 do artigo

{_tabela_artigo(raiz, "java", ablacao)}

FULL existe para os seis modelos. -CR e -TC, neste repositório, só existem para CL-13B. A Média é só de FULL.

### Rust, pass@1 do artigo

{_tabela_artigo(raiz, "rust", ablacao)}

Em CL-13B, tirar a busca (-CR) derruba o pass@1 de 52,7% para 23,3%. Tirar o contexto de tipo (-TC) deixa 49,9%.

## Contribuição

Os painéis de baixo das duas figuras mostram a amostra Java (à esquerda) e a amostra Rust (à direita), cada uma em três pares de modelos. `-CR` e `-TC` da contribuição não foram executados; os painéis mostram o sistema completo.

| Rodada | Par | Testes | Taxa | Compilação | Voltas | Grau N | Gasto |
|---|---|---|---|---|---|---|---|
{chr(10).join(blocos)}

A soma da coluna de gasto é R$ {soma_brl}. O ledger, que soma o cache da API, fecha em R$ {medido}, dentro do teto de R$ 20.

Leitura para o slide:

- **Java (`J-FF`, `gpt-6-luna` / `gpt-6-luna`, 30 tarefas):** 21 passaram em todos os testes (70,0%) e 26 compilaram (86,7%), com média de 4,3 voltas e grau N médio de 1,1 (`reasoning_effort="none"`).
- **Java (`J-JF`, `gpt-6-luna` / juíza `gpt-5`, 6 primeiras tarefas):** 5 passaram (83,3%) e 6 compilaram (100,0%), com média de 4,3 voltas e grau N médio de 1,7.
- **Java (`J-FFORTE`, `gpt-5` / `gpt-5`, 2 primeiras tarefas):** 2 passaram (100,0%) e 2 compilaram (100,0%), com média de 3,5 voltas e grau N médio de 2,0 (`reasoning_effort="minimal"`).
- **Rust (`R-FF`, `gpt-4o-mini` / `gpt-4o-mini`, 46 tarefas):** 15 passaram em todos os testes (32,6%) e 24 compilaram (52,2%). A juíza usou Continua, Reverte e Parar. A média foi 2,7 voltas e o grau N médio foi 1,7.
- **Rust (`R-JF`, `gpt-4o-mini` / juíza `gpt-5`, 8 primeiras tarefas):** 2 passaram (25,0%) e 4 compilaram (50,0%). A juíza devolveu Parar na primeira volta nas oito.
- **Rust (`R-FFORTE`, `gpt-5` / `gpt-5`, 4 primeiras tarefas):** nenhum teste passou e 3 compilaram (75,0%). Na execução original sem redução de raciocínio, a geradora consumiu os 1024 tokens de saída no raciocínio interno.

A amostra Java é um subconjunto fixo de 30 funções entre as 199 do JavaEval, e a amostra Rust é um subconjunto fixo de 46 funções entre as 90 do RustEval. O pass@1 do artigo usa o benchmark inteiro e 10 amostras independentes. Os painéis de cima e de baixo podem ir no mesmo slide, destacando essa diferença de protocolo.
"""
    (saida / "resultados.md").write_text(texto, encoding="utf-8")


def gerar(saida, motivos=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saida = Path(saida)
    raiz = saida.parents[1]
    dados = resumo(saida)
    if motivos:
        dados["motivos"] = motivos
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "resumo.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figuras = saida / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    _estilo(plt)
    general = [("In-File", "infile"), ("RepoCoder", "repocoder"), ("Our Approach", "catcoder")]
    ablacao = [("FULL", "catcoder"), ("-CR", "-cr"), ("-TC", "-tc")]
    por_id = {item["id"]: item for item in dados["rodadas"]}
    tem_java = any(por_id.get(rid, {}).get("n_completas") for rid, _, _ in PARES_JAVA)

    fig, grade = _figura(
        plt,
        "Generalização",
        "pass@1 do artigo nos painéis de cima. A contribuição entra embaixo, com outra medida.",
        "Rust do artigo, neste repositório, só tem CodeLlama-13B. "
        "A taxa de baixo é uma trajetória por tarefa na amostra, e não o pass@1.",
    )
    _painel_artigo(fig.add_subplot(grade[0, 0]), raiz, "java", general, "a", "Java")
    _painel_artigo(fig.add_subplot(grade[0, 1]), raiz, "rust", general, "b", "Rust")
    if tem_java:
        _painel_contrib(fig.add_subplot(grade[1, 0]), por_id, "c", "Contribuição em Java", PARES_JAVA)
        _painel_contrib(fig.add_subplot(grade[1, 1]), por_id, "d", "Contribuição em Rust", PARES_RUST)
    else:
        _painel_contrib(fig.add_subplot(grade[1, :]), por_id, "c", "Contribuição em Rust", PARES_RUST)
    fig.savefig(figuras / "generalizacao.png", dpi=200, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)

    fig, grade = _figura(
        plt,
        "Ablação",
        "FULL, -CR e -TC do artigo nos painéis de cima. Embaixo, o sistema completo da contribuição.",
        "-CR e -TC deste repositório só têm CodeLlama-13B. A contribuição não repetiu essas duas variantes.",
    )
    _painel_artigo(fig.add_subplot(grade[0, 0]), raiz, "java", ablacao, "c", "Java")
    _painel_artigo(fig.add_subplot(grade[0, 1]), raiz, "rust", ablacao, "d", "Rust")
    if tem_java:
        _painel_contrib(fig.add_subplot(grade[1, 0]), por_id, "e", "Contribuição em Java, sistema completo", PARES_JAVA)
        _painel_contrib(fig.add_subplot(grade[1, 1]), por_id, "f", "Contribuição em Rust, sistema completo", PARES_RUST)
    else:
        _painel_contrib(fig.add_subplot(grade[1, :]), por_id, "e", "Contribuição em Rust, sistema completo", PARES_RUST)
    fig.savefig(figuras / "ablacao.png", dpi=200, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    ledger = saida / "ledger.json"
    if ledger.is_file():
        dados["medido_brl"] = json.loads(ledger.read_text(encoding="utf-8")).get("medido_brl")
    _descrever(saida, raiz, dados, _ler_tarefas(saida))
    return dados


def _amostras_doc():
    assert RUST_IDX[:4] == [8, 7, 29, 44]
    assert JAVA_IDX[:2] == [92, 59]
    assert RUST_IDX[:8][0] == 8
    total = sum(reserva_brl(r["geradora"], r["juiza"]) * len(r["indices"]) for r in RODADAS)
    assert total <= TETO_BRL
    assert total > 18.5
