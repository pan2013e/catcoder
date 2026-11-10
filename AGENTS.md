# AGENTS.md — CatCoder (fork do grupo)

Leia este arquivo antes de alterar o repositório. Ele fixa o CatCoder do artigo, os commits do fork, a arquitetura da contribuição e o que dessa arquitetura já está na árvore de `dev-davi`. Mudanças futuras partem daqui. Não reimplemente o que o artigo já tem. O laço Gemini de `47e841e` não é mais o código de trabalho: em `dev-davi` ele foi substituído pelo ciclo em `catcoder/contribuicao/`, ainda sem commit.

## O que é este repositório

Fork de **CATCODER: Repository-Level Code Generation with Relevant Code and Type Context** (Zhiyuan Pan). O artigo gera o corpo de uma função a partir de dois contextos do repositório: código relevante recuperado por busca híbrida e contexto de tipo extraído por análise estática. A avaliação é pass@k (amostras independentes, compilação e testes).

Remote: `https://github.com/AmandaCLima/catcoder.git`. A `main` local e a `origin/main` estão em `47e841e`. O trabalho do grupo nesta máquina está em `dev-davi`, que rastreia `origin/dev-davi` em `c8260a8`. O ciclo, o grau N, os resultados e as figuras estão na árvore de trabalho, ainda sem commit.

Histórico, do mais antigo ao atual:

| Commit | Autor | O que é |
|---|---|---|
| `3be2687` upload | Zhiyuan Pan | Código, datasets e resultados do artigo |
| `1e35a5e`, `6c3874e` | Zhiyuan Pan | Ajustes do README. **`6c3874e` é o último commit do artigo** |
| `1cd0182` | AmandaCLima (2026-11-06) | Pipeline Rust executável na máquina local |
| `47e841e` | AmandaCLima (2026-11-09) | Loop gerador + juiz Gemini no Rust. **`main`** |
| `c8260a8` | Davi-SB (2026-11-10) | `AGENTS.md` e `.gitignore`. **Ponta commitada de `dev-davi`** |

Java do artigo, a recuperação híbrida e o template do prompt continuam como no artigo. A contribuição mora em `catcoder/contribuicao/`, com o grau N no extrator Rust.

## Layout

```
README.md                         instruções originais (Linux, GPU, Defects4J, rust-analyzer)
docs/arquitetura.md               desenho da contribuição e tetos
docs/experimentos.md              amostras, preços, teto de R$ 20 e o que é copiado do artigo
results/{java,rust}/...           números e códigos gerados do artigo; não sobrescrever
results/contribuicao/             amostra local: tarefas, cache, ledger, figuras, resultados.md
catcoder/requirements.txt         deps do experimento (vLLM, LangChain, Chroma, rank_bm25, datasets)
catcoder/contribuicao/            ciclo, runner, cache, figuras. Entrada: python -m catcoder.contribuicao.run
catcoder/java/                    pipeline Java do artigo, intacto
catcoder/java/dataset/javaeval/   metadados do JavaEval; o Arrow data-00000-of-00001.arrow não está no disco
catcoder/rust/                    pipeline Rust; evaluation.py agora só aponta para o runner
catcoder/rust/dataset/rusteval/   HuggingFace dataset já com contextos pré-calculados (90 tarefas)
catcoder/rust/crates/             crates do benchmark, descompactados em 1cd0182
catcoder/rust/prompts/codegen     template [CONTEXT] + [CODE]
catcoder/rust/results/            saídas locais do Gemini (não são os resultados do artigo)
catcoder/tools/intellirust/       análise estática Rust; Context.build recebe degree
catcoder/tools/intellirust/vendor/ ra_ap_stdx 0.0.198 com o cast que o rustc atual recusa
catcoder/tools/java/              análise estática Java (Eclipse JDT LS via multilspy)
```

Os JSON em `results/java` e `results/rust` são do artigo. A amostra da contribuição fica em `results/contribuicao/` e não entra na média do pass@1. O `evaluation.py` do Rust, na árvore de trabalho, não gera mais: encerra e manda usar `python -m catcoder.contribuicao.run` a partir da raiz.

Crates do RustEval: `bit-vec`, `curv-kzen`, `dlv-list`, `gptman`, `mathru`, `multiset`, `rocket_http`, `sdl2`, `splay_tree`, `sshkeys`, `unix_path`, `vec_map`, `without-alloc`. Há um `crates.zip` aninhado em `bit-vec/`; é resíduo da descompactação, não é fonte.

## Pipeline original do artigo

O que segue descreve `6c3874e`. O grau N do Rust, na árvore de `dev-davi`, está parametrizado; o restante desta seção continua valendo.

Geração em uma passada. Para cada tarefa, o modelo produz `n` amostras independentes (padrão `n=10`, temperatura 0.6, top_p 0.7). Cada amostra vê o mesmo prompt. Não há revisão, juiz nem escolha de versão. A métrica é compile@k e pass@k (`catcoder/rust/metrics.py`, `CratePassK`; o estimador é o de Chen et al., 2021).

O prompt está em `catcoder/rust/prompts/codegen` (o de Java é o equivalente):

```
[CONTEXT]
{{ focal_ctx }}
{{ rag_data }}
[/CONTEXT]
[CODE]
{{ docstring }}
{{ focal_fn_signature }}
```

`build_prompt` em `catcoder/{java,rust}/util.py` renderiza esse template. A inferência original está em `inference.py`: `OpenAIModel` (gpt-3.5 / gpt-4) ou `VllmModel` / `VllmClientModel`. O `__main__` do artigo usa `VllmClientModel('codellama-13b')`.

### De onde vem cada campo do prompt

Os contextos **não** são calculados na avaliação. `evaluation.py` só lê o dataset. Quem produz os campos:

| Campo no dataset | Quem produz | O que entra no prompt |
|---|---|---|
| `rag_data` | `retrieve_relevant_code.py` → `run_rag` | código relevante |
| `extended_context` | `extract_type_context.py` → `make_ctx` | contexto de tipo |
| `repocoder_data` | `repocoder_rag` | baseline RepoCoder |
| `infile_context` | pré-calculado no dataset | baseline in-file |
| `signature`, `docstring`, `hint`, `fn_full`, `lines`, `package`, `path` | benchmark | função focal e local da edição |

Busca híbrida, já implementada nos dois idiomas (`RustProjectIndexer` / o indexador Java):

- denso: HuggingFace embeddings + Chroma, `k=8`, cache em `./.rag_cache` e `./.embedding_cache`
- esparso: `BM25Retriever`, `k=8`
- fusão: `EnsembleRetriever` com pesos **0.7 denso, 0.3 BM25**
- a função focal é removida do índice (`drop_ground_truth`) para não vazar o corpo verdadeiro
- a query é `hint + assinatura`

Contexto de tipo, profundidade **fixa no código**, sem parâmetro de grau:

- Rust, `catcoder/tools/intellirust/intellirust/context.py`, `Context.build`: tipos da assinatura via rust-analyzer e mais um salto para tipos internos (`query_typedef`). `StringBuilder` serializa dois níveis. `TreeNode.filter_by_type` poda impls com `prune_forest(..., max_levels=1)`. Tipos da stdlib são ignorados depois do primeiro elemento.
- Java, `catcoder/tools/java/java_analyzer/analyzer.py`, `Analyzer.build_context`: classe envolvente, tipos dos parâmetros, tipo de retorno e tipos dos campos. Sem caminhada configurável.

`TreeNode.prune_forest` (`file_structure.py`) é o único podador de grafo/árvore que existe. `max_levels == 0` devolve os nós como estão; `1` filtra a raiz e mantém os filhos.

### Benchmarks no `evaluation.py` original

`6c3874e:catcoder/rust/evaluation.py` (Java espelha, e o Java **ainda** tem estas classes):

| Classe | Nome do resultado | `focal_ctx` | `rag_data` |
|---|---|---|---|
| `RustEvalCatCoder` | `rusteval_xc` | `extended_context` | `rag_data` |
| `RustEvalInFile` | `rusteval_if` | `infile_context` | vazio |
| `RustEvalRepoCoder` | `rusteval_repo` | vazio | `repocoder_data` |
| `RustEvalVanilla` | `rusteval_basic` | vazio | vazio |
| `RustEvalWithoutContext` | `rusteval-tc` | vazio | `rag_data` |
| `RustEvalWithoutRetrieval` | `rusteval-cr` | `extended_context` | vazio |

O teste Rust copia o crate para um diretório temporário, substitui o corpo da função focal (`TestAdapter`, `replace_test=False`) e roda `cargo build` e `cargo test --doc`. Java usa Defects4J.

## Estado atual do fork

### `1cd0182` — pipeline local

Três arquivos de lógica, mais a árvore `catcoder/rust/crates/` descompactada:

- `metrics.py`: `multiprocessing.Pool(50)` virou laço sequencial. Motivo registrado no código: `WinError 1450` e estouro de RAM no Windows.
- `test_adapter.py`: `proc.wait` virou `communicate`, timeout de 30s para 120s, com `kill` + `communicate` no timeout. Motivo: deadlock do pipe do `cargo`.
- `evaluation.py`: trocou vLLM/OpenAI por um `GeminiModel` (`google-genai`, `models/gemini-3.1-flash-lite`, temperatura 0.2). Cortou o dataset para 5 tarefas. Apagou deste arquivo as classes de baseline e deixou `Benchmark`, `RustEval` e `RustEvalCatCoder`. A geração continuou em uma passada, e `CratePassK` ainda media compile@k / pass@k.

README do artigo continua pedindo Linux e GPU. O fork passou a rodar no Windows, em sequência, com Gemini.

### `47e841e` — loop gerador + juiz (main)

Este commit é a `main`. Não é o código de trabalho em `dev-davi`: lá `evaluation.py` virou um aviso para o runner. O comportamento abaixo fica como referência do que a Amanda entregou, para não ser reintroduzido no lugar do ciclo.

`catcoder/rust/evaluation.py` foi reescrito. A hierarquia `Benchmark` / `RustEval*` saiu deste arquivo. O fluxo atual de `gerar_e_testar_codigo`:

1. Monta o prompt uma vez com `build_prompt` sobre `signature`, `docstring`, `extended_context`, `rag_data`. Recuperação e extração de tipo não rodam aqui; usam o dataset pré-calculado.
2. Até 10 iterações. A primeira manda o prompt do artigo. As seguintes acrescentam o código anterior e o último feedback, e pedem só código Rust.
3. **LLM geradora** (`gemini-3.1-flash-lite`, temperatura 0.2) devolve código. Há limpeza de markdown e um corte heurístico até `pub fn` / `fn`.
4. `TestAdapter.compile()` (`cargo build`). `adapter.test()` não é chamado neste loop.
5. **LLM juíza** (mesmo modelo, temperatura 0.1, JSON) devolve `{"nota", "feedback_critico"}`.
6. Guarda o código se a nota subir. Para se compilou e a nota é >= 9. Senão, espera 4s e repete.
7. `main` corta o dataset em 10 tarefas e grava em `results/rusteval_xc/models-gemini-3.1-flash-lite_n10.json`. O vencedor é repetido 10 vezes em `fn_codes` para imitar o formato `n=10`. Não há `lastest_eval` nem pass@k.

Comportamento real desse juiz, para não ser confundido com o desenho alvo:

- A juíza emite nota e texto. Não emite Continua, Reverte ou Parar, e não escolhe uma versão entre um conjunto.
- A parada é um limiar no Python (`compilou and nota >= 9`) ou o fim das 10 iterações.
- `codigo_anterior` só muda quando a nota sobe, mas o feedback injetado no próximo prompt é sempre o último, mesmo quando o código guardado é o de uma iteração anterior.
- A chave vem de `GEMINI_API_KEY`, com um literal placeholder se a variável falta. Não colocar chave real no arquivo.

Java, `retrieve_relevant_code.py`, `extract_type_context.py`, `intellirust` e o template `codegen` não foram alterados por nenhum dos dois commits.

## Arquitetura alvo

A especificação da contribuição está em `docs/arquitetura.md`. O plano de experimentos, modelos e o teto de R$ 20 estão em `docs/experimentos.md`. O fluxo abaixo é o resumo para orientar mudanças; a arquitetura é a referência.

```
Recuperação de código relevante
  Busca híbrida (BM25 + denso)
        |
        v
Define o grau N                         ramo paralelo
  Usuário ou LLM decide
        |
        v
Extração de contexto de tipo
  Grafo podado até o grau N
        |
        +------------------+
                           v
                  Construção do prompt
                  [CONTEXT] + [CODE]
                           |
                           v
                  LLM geradora
                  Gera ou revisa o código
                           |
                           v
                  LLM juíza
                  Escolhe qual versão reavaliar
                    |-- Continua  (seta cheia)  --> volta à geradora
                    |-- Reverte   (seta tracejada) --> volta à geradora
                    +-- Parar -----------------------> Código final entregue
```

Mapeamento para o código da árvore de `dev-davi`:

| Etapa do desenho | Onde está | Situação |
|---|---|---|
| Busca híbrida BM25 + denso | `retrieve_relevant_code.py`, `EnsembleRetriever` 0.7/0.3 | Pronta no artigo. O ciclo lê `rag_data` do dataset e corta em 16 trechos de 1 000 caracteres. Não reindexa |
| Grau N decidido por LLM | `ciclo.escolher_n`, `gpt-4o-mini`, teto 4 | Feito. Uma chamada por tarefa, cache compartilhado entre as rodadas da mesma língua |
| Grafo de tipos podado até N | `Context.build(degree=)`, `make_ctx` | Feito no Rust. N=0 devolve vazio; o padrão 2 preserva o salto do artigo. Java não tem esse grafo |
| Prompt `[CONTEXT]` + `[CODE]` | `prompts/codegen`, `util.build_prompt` | Igual ao artigo |
| LLM geradora, gera ou revisa | `ciclo.trajetoria` | Feito. Da segunda volta em diante revisa uma versão e o feedback |
| LLM juíza escolhe a versão | `ciclo.trajetoria` | Feito. JSON `{"acao","versao","feedback"}`: Continua, Reverte ou Parar |
| Continua / Reverte | conjunto `versoes` no ciclo | Feito. Continua revisa a versão recém-gerada. Reverte revisa o id anterior que a juíza indicar |
| Parar → código final | fim do ciclo, depois `TestAdapter.test` | Feito. Parar, ou o fim da 5ª volta, entrega essa versão. O teste roda uma vez, no código entregue |
| Métrica | `results/contribuicao/resumo.json` | Uma trajetória por tarefa. Sucesso = compilou e passou em todos os testes. Não é pass@k e não repete o vencedor 10 vezes |

## O que já foi feito

Tudo isto está na árvore de `dev-davi` e ainda não foi commitado, fora o `c8260a8`.

- Pacote `catcoder/contribuicao/` com tetos, cache por chamada, ledger de R$ 20 e o ciclo Continua / Reverte / Parar. `selfcheck.py` cobre corte, teto e retomada com cliente falso.
- `Context.build` e `make_ctx` recebem `degree`. O texto de tipo fica em `results/contribuicao/cache_tipo/`.
- `intellirust` compila no rustc atual com `vendor/ra_ap_stdx` e o `[patch.crates-io]` em `Cargo.toml`. O `.pyd` gerado não entra no git.
- `TestAdapter` do Rust abre o fonte do crate em UTF-8. Sem isso, no Windows, a leitura quebra em cp1252.
- Ambiente desta máquina: Python 3.12, `openai`, `datasets`, `matplotlib`, cargo e rust-analyzer. A chave é `OPENAI_API_KEY` no `.env`, que continua fora do git.
- Três rodadas Rust, em 10 nov 2026, gasto medido no ledger R$ 1,11 de um teto de R$ 20. Os números estão em `results/contribuicao/resultados.md`.

| Rodada | Par | Amostra | Passou nos testes | Compilou |
|---|---|---|---|---|
| R-FF | gpt-4o-mini / gpt-4o-mini | 46 | 15 (32,6%) | 24 (52,2%) |
| R-JF | gpt-4o-mini / gpt-5 | 8 primeiras | 2 (25%) | 4 (50%) |
| R-FFORTE | gpt-5 / gpt-5 | 4 primeiras | 0 | 3 (75%) |

Cinco tarefas (R-FF 47, 48, 75, 77 e R-JF 75) pararam na primeira execução na leitura do fonte. Foram concluídas com `python -X utf8` e o cache da API. Os números da tabela já incluem essa retomada.

As figuras `results/contribuicao/figuras/generalizacao.png` e `ablacao.png` copiam o pass@1 dos JSON do artigo e acrescentam a taxa da amostra Rust num painel separado. A Média do artigo só entra quando os seis modelos existem. Rust do artigo, neste checkout, só tem CodeLlama-13B. Java de -CR e -TC também só tem CodeLlama-13B.

## O que ainda não

- **Java da contribuição.** J-FF (30), J-JF (6) e J-FFORTE (2) não começaram. O preflight exige `defects4j` no PATH, e ele não está instalado. Além disso falta `catcoder/java/dataset/javaeval/data-00000-of-00001.arrow`, que o `state.json` nomeia. Defects4J é um fluxo de Linux (Perl, Git, SVN, JDK 8 e 11); nesta máquina o caminho é o WSL, com o experimento rodando dentro dele. A reserva dessas rodadas não foi gasta em tarefas Rust a mais.
- **Poda por grau no Java.** O `Analyzer.build_context` do artigo não caminha num grafo. A rodada Java, quando existir, usa o `extended_context` do dataset, cortado em 12 000 caracteres, e só registra o N pedido.
- **Barras do slide sem JSON.** Modelos de Rust que não são CodeLlama-13B, e -CR/-TC dos outros modelos de Java, ficam em branco. Não ler esses valores no screenshot.
- **Ablação da contribuição.** -CR e -TC não foram reexecutados. O painel novo é só o sistema completo.
- **Par gpt-5 / gpt-5.** A geradora devolveu código vazio e consumiu os 1024 tokens de saída em raciocínio interno. R-FFORTE em 0% não é uma comparação limpa de qualidade do modelo. Subir o teto de saída do gpt-5, se a conta couber nos R$ 20, é o conserto.
- **Commit.** Código, `docs/`, resultados e figuras ainda não foram commitados em `dev-davi`. Nada disso entra na `main`.

Retomar a amostra usa o mesmo comando, a partir da raiz, com `PYTHONUTF8=1` no Windows. Tarefa com `completo` é pulada. Chamada presente no cache não é repetida e não soma de novo no ledger.

## Restrições ao modificar

- Superfície ativa: `catcoder/contribuicao/`, o `degree` em `catcoder/tools/intellirust/` e `make_ctx` em `catcoder/rust/extract_type_context.py`. Java permanece o pipeline do artigo até a rodada Java ser possível.
- Preservar `6c3874e` como referência do artigo. As classes de baseline do `evaluation.py` Rust estão em `catcoder/java/evaluation.py` e nesse commit. O `evaluation.py` Rust da árvore de trabalho não volta a ser o laço Gemini.
- Não sobrescrever `results/java` nem `results/rust`. Não apresentar `results/contribuicao/` nem `catcoder/rust/results/` como resultado do artigo.
- Não commitar chave de API. A contribuição lê `OPENAI_API_KEY` do `.env`. O laço de `47e841e` lia `GEMINI_API_KEY`.
- Manter o que `1cd0182` consertou para o Windows: `cargo` com `communicate` e timeout, avaliação de crates em sequência. Um `Pool` de dezenas de `cargo` reabre o `WinError 1450`. A leitura do fonte do crate fica em UTF-8, e o comando roda com `python -X utf8`.
- Dependência nova só se a stdlib e o que está em `catcoder/requirements.txt` não cobrirem. O ciclo usa `openai` e `python-dotenv`; as figuras usam `matplotlib`. Nenhum dos três está no `requirements.txt` do artigo.
- Comentários novos no código do fork podem seguir o português dos commits do grupo. Identificadores e o prompt do artigo ficam como estão.
- Não duplicar `EnsembleRetriever` nem o template Jinja. O ciclo chama `build_prompt`.
- Os tetos de voltas, tokens de saída, grau N e tamanho de prompt estão em `docs/arquitetura.md`, seção Decisões. A amostra e o teto de R$ 20 estão em `docs/experimentos.md`. Implementação corta para esses tetos. Chamada de API sem `max_output_tokens` não sai. No gpt-5 esse teto inclui o raciocínio interno.
