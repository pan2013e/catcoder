# AGENTS.md — CatCoder (fork do grupo)

Leia este arquivo antes de alterar o repositório. Ele fixa três estados: o CatCoder do artigo, o que os dois commits do fork já fizeram, e a arquitetura alvo da contribuição. Mudanças futuras partem daqui. Não reimplemente o que o artigo já tem e não trate o loop atual como se ele já fosse o desenho alvo.

## O que é este repositório

Fork de **CATCODER: Repository-Level Code Generation with Relevant Code and Type Context** (Zhiyuan Pan). O artigo gera o corpo de uma função a partir de dois contextos do repositório: código relevante recuperado por busca híbrida e contexto de tipo extraído por análise estática. A avaliação é pass@k (amostras independentes, compilação e testes).

Remote: `https://github.com/AmandaCLima/catcoder.git`, branch `main`.

Histórico, do mais antigo ao atual:

| Commit | Autor | O que é |
|---|---|---|
| `3be2687` upload | Zhiyuan Pan | Código, datasets e resultados do artigo |
| `1e35a5e`, `6c3874e` | Zhiyuan Pan | Ajustes do README. **`6c3874e` é o último commit do artigo** |
| `1cd0182` | AmandaCLima (2026-11-06) | Pipeline Rust executável na máquina local |
| `47e841e` | AmandaCLima (2026-11-09) | Loop gerador + juiz no Rust. **HEAD** |

Java, ferramentas de análise e os scripts de recuperação/extração continuam no estado do artigo. A contribuição em curso está no pipeline Rust.

## Layout

```
README.md                         instruções originais (Linux, GPU, Defects4J, rust-analyzer)
results/{java,rust}/...           números e códigos gerados do artigo; não sobrescrever
catcoder/requirements.txt         deps do experimento (vLLM, LangChain, Chroma, rank_bm25, datasets)
catcoder/java/                    pipeline Java do artigo, intacto
catcoder/rust/                    pipeline Rust; superfície ativa do fork
catcoder/rust/dataset/rusteval/   HuggingFace dataset já com contextos pré-calculados
catcoder/rust/crates/             crates do benchmark, descompactados em 1cd0182
catcoder/rust/prompts/codegen     template [CONTEXT] + [CODE]
catcoder/rust/results/            saídas locais do Gemini (não são os resultados do artigo)
catcoder/tools/intellirust/       análise estática Rust (rust-analyzer)
catcoder/tools/java/              análise estática Java (Eclipse JDT LS via multilspy)
```

O `evaluation.py` grava em caminho relativo. Rodado de `catcoder/rust`, a saída cai em `catcoder/rust/results/`. Os JSON em `results/` na raiz são do artigo.

Crates do RustEval: `bit-vec`, `curv-kzen`, `dlv-list`, `gptman`, `mathru`, `multiset`, `rocket_http`, `sdl2`, `splay_tree`, `sshkeys`, `unix_path`, `vec_map`, `without-alloc`. Há um `crates.zip` aninhado em `bit-vec/`; é resíduo da descompactação, não é fonte.

## Pipeline original do artigo

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

### `47e841e` — loop gerador + juiz (HEAD)

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

A contribuição muda o CatCoder para o fluxo abaixo. Recuperação e extração seguem em paralelo e se encontram na construção do prompt; daí para baixo é um ciclo geradora/juíza.

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

Mapeamento para o código que já existe:

| Etapa do desenho | Onde encostar | Situação em HEAD |
|---|---|---|
| Busca híbrida BM25 + denso | `retrieve_relevant_code.py`, `EnsembleRetriever` 0.7/0.3 | Pronta no artigo. A avaliação consome `rag_data` pronto e não chama o retriever |
| Grau N decidido por usuário ou LLM | não existe | Profundidade fixa em `Context.build` e `Analyzer.build_context` |
| Grafo de tipos podado até N | `TreeNode.prune_forest`, `Context.build`, `extract_type_context.py` | Poda existe com `max_levels` cravado (1 no filtro de impl; um salto extra de tipo no `build`) |
| Prompt `[CONTEXT]` + `[CODE]` | `prompts/codegen`, `util.build_prompt` | Igual ao artigo. Manter esse formato |
| LLM geradora, gera ou revisa | `evaluation.py`, `gerar_e_testar_codigo` | Gera e, da segunda iteração em diante, revisa com código + feedback |
| LLM juíza escolhe a versão | `evaluation.py`, segunda chamada Gemini | Só dá nota. Não escolhe versão nem ação |
| Continua | seta cheia de volta à geradora | O laço sempre continua com o último feedback |
| Reverte | seta tracejada de volta à geradora | Não há ação de reverter. Há um "melhor até agora" atualizado só quando a nota sobe |
| Parar → código final | saída do laço | Limiar compilou e nota >= 9, ou 10 iterações |

## O que falta para o desenho

Planejar em cima destes buracos, nesta ordem. Cada item é uma diferença entre HEAD e a figura.

1. **Grau N.** Expor a profundidade do grafo de tipos. Quem decide é o usuário ou uma LLM, antes da extração. O valor entra na poda. Hoje o dataset guarda um `extended_context` de profundidade fixa; mudar N exige reextrair (chamar `make_ctx` / `Context.build` no laço, ou regenerar o campo), porque ler `extended_context` ignora N.
2. **Poda até N.** Parametrizar a caminhada em `Context.build` e o `max_levels` de `prune_forest`, em vez de criar outro extrator. Java fica como está até a tarefa pedir paridade.
3. **Decisão da juíza.** A saída deixa de ser só `{nota, feedback}`. A juíza escolhe qual versão reavaliar e devolve uma ação: Continua (revisar a versão corrente), Reverte (voltar a uma versão anterior e reavaliar essa), Parar (entregar o código final). As duas primeiras voltam para a geradora; Parar encerra.
4. **Memória de versões.** Continua e Reverte precisam de um conjunto de versões (código, se compilou, feedback). O par `melhor_codigo` / `codigo_anterior` não representa essa escolha.
5. **Métrica.** O artigo reporta compile@k e pass@k com `n` amostras independentes. O laço alvo produz uma trajetória por tarefa. Não gravar o vencedor repetido 10 vezes e chamar isso de pass@k. Quando for medir de novo, `CratePassK` e `TestAdapter.test` continuam em `metrics.py` / `test_adapter.py`; o laço atual só chama `compile()`.

Recuperação híbrida e o template `[CONTEXT]` + `[CODE]` já batem com a figura. O trabalho neles é ligá-los ao laço se N e a extração passarem a acontecer em tempo de geração, não reescrevê-los.

## Restrições ao modificar

- Superfície ativa: `catcoder/rust/` e, se o grau N precisar sair do extrator, `catcoder/tools/intellirust/`. Java permanece o pipeline do artigo até alguém pedir o contrário.
- Preservar `6c3874e` como referência do artigo. As classes de baseline removidas do `evaluation.py` Rust ainda estão em `catcoder/java/evaluation.py` e nesse commit.
- Não sobrescrever `results/` da raiz nem apresentar `catcoder/rust/results/` como resultado do artigo.
- Não commitar chave de API. `GEMINI_API_KEY` no ambiente.
- Manter o que `1cd0182` consertou para o Windows: `cargo` com `communicate` e timeout, avaliação de crates em sequência. Um `Pool` de dezenas de `cargo` reabre o `WinError 1450`.
- Dependência nova só se a stdlib e o que está em `catcoder/requirements.txt` não cobrirem. Gemini entrou via `google-genai`, que não está no `requirements.txt` do artigo.
- Comentários novos no código do fork podem seguir o português dos dois commits. Identificadores e o prompt do artigo ficam como estão, salvo a tarefa que muda o contrato da juíza.
- Diff mínimo no lugar certo: profundidade de tipo no extrator, decisão do ciclo em `evaluation.py`. Não duplicar `EnsembleRetriever` nem o template Jinja.
