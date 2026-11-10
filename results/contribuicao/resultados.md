# Resultados para os slides

Os painéis do artigo repetem o recorte dos slides: In-File, RepoCoder e Our Approach na generalização; FULL, -CR e -TC na ablação. As cores são as da legenda daqueles slides. A ordem dos modelos em Java é CG-7B, ML-8B, Média, CL-7B, CQ-7B, DS-6.7B, CL-13B. Em Rust, a ordem do slide seria ML-8B, CG-7B, CL-7B, Média, DS-6.7B, CQ-7B, CL-13B.

Cada número do artigo é o `pass@1` de `lastest_eval` no JSON de `results/`. A Média só entra quando os seis modelos existem. Modelo sem JSON fica de fora da figura e da tabela, em branco. Nada foi lido a partir do screenshot.

A contribuição fica no painel de baixo, com outro eixo. A taxa é a fração de tarefas da amostra em que o código entregue compilou e passou em todos os testes. É uma trajetória por tarefa. Esse número não é o pass@1 do artigo e não entra na Média.

## Generalização

Arquivo: `figuras/generalizacao.png`.

### Java, pass@1 do artigo

| Modelo | In-File | RepoCoder | Our Approach |
|---|---|---|---|
| CG-7B | 19,8% | 30,2% | 33,3% |
| ML-8B | 18,8% | 29,2% | 34,5% |
| Média | 27,0% | 34,7% | 39,4% |
| CL-7B | 30,3% | 35,6% | 39,6% |
| CQ-7B | 30,7% | 37,2% | 42,0% |
| DS-6.7B | 26,8% | 34,9% | 42,4% |
| CL-13B | 35,3% | 41,0% | 44,7% |

O ganho de Our Approach sobre In-File fica entre 13 e 16 pontos em CG-7B, ML-8B e DS-6.7B, e perto de 9 pontos em CL-7B e CL-13B. Sobre RepoCoder, o ganho fica entre 3 e 8 pontos.

### Rust, pass@1 do artigo

| Modelo | In-File | RepoCoder | Our Approach |
|---|---|---|---|
| CL-13B | 41,6% | 49,4% | 52,7% |

Neste repositório o Rust do artigo só tem CodeLlama-13B. Os outros modelos do slide ficam em branco.

## Ablação

Arquivo: `figuras/ablacao.png`.

### Java, pass@1 do artigo

| Modelo | FULL | -CR | -TC |
|---|---|---|---|
| CG-7B | 33,3% | — | — |
| ML-8B | 34,5% | — | — |
| Média | 39,4% | — | — |
| CL-7B | 39,6% | — | — |
| CQ-7B | 42,0% | — | — |
| DS-6.7B | 42,4% | — | — |
| CL-13B | 44,7% | 27,5% | 39,5% |

FULL existe para os seis modelos. -CR e -TC, neste repositório, só existem para CL-13B. A Média é só de FULL.

### Rust, pass@1 do artigo

| Modelo | FULL | -CR | -TC |
|---|---|---|---|
| CL-13B | 52,7% | 23,3% | 49,9% |

Em CL-13B, tirar a busca (-CR) derruba o pass@1 de 52,7% para 23,3%. Tirar o contexto de tipo (-TC) deixa 49,9%.

## Contribuição

O painel de baixo das duas figuras. As três barras são a mesma amostra Rust, em três pares. Java não tem barra: Defects4J não está instalado, então J-FF, J-JF e J-FFORTE não começaram. -CR e -TC da contribuição também não foram executados. O painel mostra o sistema completo.

| Par | Testes | Taxa | Compilação | Voltas | Grau N | Gasto |
|---|---|---|---|---|---|---|
| 4o-mini / juíza 4o-mini | 15 de 46 | 32,6% | 24 de 46 | 2,7 | 1,7 | R$ 0,40 |
| 4o-mini / juíza gpt-5 | 2 de 8 | 25,0% | 4 de 8 | 1,0 | 2,2 | R$ 0,21 |
| gpt-5 / juíza gpt-5 | 0 de 4 | 0,0% | 3 de 4 | 1,0 | 2,5 | R$ 0,38 |

A soma da coluna de gasto é R$ 0,99. O ledger, que soma o cache da API, fecha em R$ 1,11, dentro do teto de R$ 20. A diferença está nas cinco tarefas refeitas: o custo delas já estava no cache e não voltou para o arquivo da tarefa.

Leitura para o slide:

- 4o-mini nos dois papéis, 46 tarefas: 15 passaram em todos os testes (32,6%) e 24 compilaram (52,2%). A juíza usou Continua, Reverte e Parar. A média foi 2,7 voltas e o grau N médio foi 1,7.
- 4o-mini com juíza gpt-5, nas 8 primeiras tarefas: 2 passaram (25%) e 4 compilaram. A juíza devolveu Parar na primeira volta nas oito.
- gpt-5 nos dois papéis, nas 4 primeiras: nenhum teste passou e 3 compilaram. A geradora devolveu código vazio e consumiu os 1024 tokens de saída. O teto foi gasto em raciocínio interno do modelo, e não sobrou texto para o corpo da função.

Cinco tarefas tinham parado na primeira execução porque o Windows leu o fonte do crate fora de UTF-8 (R-FF 47, 48, 75, 77 e R-JF 75). Foram concluídas depois, com Python em modo UTF-8, reusando o cache da API. Os números acima já incluem essa retomada.

A amostra Rust é um subconjunto fixo de 46 funções entre as 90 do RustEval. O pass@1 do artigo usa o benchmark inteiro e 10 amostras independentes. Os dois painéis podem ir no mesmo slide, com essa diferença dita em voz alta.
