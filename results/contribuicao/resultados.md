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

Os painéis de baixo das duas figuras mostram a amostra Java (à esquerda) e a amostra Rust (à direita), cada uma em três pares de modelos. `-CR` e `-TC` da contribuição não foram executados; os painéis mostram o sistema completo.

| Rodada | Par | Testes | Taxa | Compilação | Voltas | Grau N | Gasto |
|---|---|---|---|---|---|---|---|
| J-FF (Java) | gpt-6-luna / juíza gpt-6-luna | 21 de 30 | 70,0% | 26 de 30 (86,7%) | 4,3 | 1,1 | R$ 0,45 |
| J-JF (Java) | gpt-6-luna / juíza gpt-5 | 5 de 6 | 83,3% | 6 de 6 (100,0%) | 4,3 | 1,7 | R$ 0,68 |
| J-FFORTE (Java) | gpt-5 / juíza gpt-5 | 2 de 2 | 100,0% | 2 de 2 (100,0%) | 3,5 | 2,0 | R$ 0,40 |
| R-FF (Rust) | 4o-mini / juíza 4o-mini | 15 de 46 | 32,6% | 24 de 46 (52,2%) | 2,7 | 1,7 | R$ 0,40 |
| R-JF (Rust) | 4o-mini / juíza gpt-5 | 2 de 8 | 25,0% | 4 de 8 (50,0%) | 1,0 | 2,2 | R$ 0,21 |
| R-FFORTE (Rust) | gpt-5 / juíza gpt-5 | 0 de 4 | 0,0% | 3 de 4 (75,0%) | 1,0 | 2,5 | R$ 0,38 |

A soma da coluna de gasto é R$ 2,51. O ledger, que soma o cache da API, fecha em R$ 2,65, dentro do teto de R$ 20.

Leitura para o slide:

- **Java (`J-FF`, `gpt-6-luna` / `gpt-6-luna`, 30 tarefas):** 21 passaram em todos os testes (70,0%) e 26 compilaram (86,7%), com média de 4,3 voltas e grau N médio de 1,1 (`reasoning_effort="none"`).
- **Java (`J-JF`, `gpt-6-luna` / juíza `gpt-5`, 6 primeiras tarefas):** 5 passaram (83,3%) e 6 compilaram (100,0%), com média de 4,3 voltas e grau N médio de 1,7.
- **Java (`J-FFORTE`, `gpt-5` / `gpt-5`, 2 primeiras tarefas):** 2 passaram (100,0%) e 2 compilaram (100,0%), com média de 3,5 voltas e grau N médio de 2,0 (`reasoning_effort="minimal"`).
- **Rust (`R-FF`, `gpt-4o-mini` / `gpt-4o-mini`, 46 tarefas):** 15 passaram em todos os testes (32,6%) e 24 compilaram (52,2%). A juíza usou Continua, Reverte e Parar. A média foi 2,7 voltas e o grau N médio foi 1,7.
- **Rust (`R-JF`, `gpt-4o-mini` / juíza `gpt-5`, 8 primeiras tarefas):** 2 passaram (25,0%) e 4 compilaram (50,0%). A juíza devolveu Parar na primeira volta nas oito.
- **Rust (`R-FFORTE`, `gpt-5` / `gpt-5`, 4 primeiras tarefas):** nenhum teste passou e 3 compilaram (75,0%). Na execução original sem redução de raciocínio, a geradora consumiu os 1024 tokens de saída no raciocínio interno.

A amostra Java é um subconjunto fixo de 30 funções entre as 199 do JavaEval, e a amostra Rust é um subconjunto fixo de 46 funções entre as 90 do RustEval. O pass@1 do artigo usa o benchmark inteiro e 10 amostras independentes. Os painéis de cima e de baixo podem ir no mesmo slide, destacando essa diferença de protocolo.
