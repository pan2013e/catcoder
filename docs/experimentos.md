# Experimentos

Plano do que será executado na API da OpenAI e do que será só copiado do artigo. O teto de gasto de todas as rodadas juntas é **R$ 20**. A conta abaixo, no pior caso de tokens, fecha em **R$ 19,99**.

Os experimentos do artigo não são reexecutados. As barras de In-File, RepoCoder, Our Approach, FULL, -CR e -TC saem dos JSON em `results/` ou, quando o JSON não está neste checkout, do valor publicado no artigo. A parte nova é a contribuição: grau N escolhido por LLM, busca híbrida, prompt `[CONTEXT]` + `[CODE]` e o ciclo geradora/juíza. Essa parte roda em amostra.

## Preço usado na conta

Tabela oficial da OpenAI, consultada em 10 nov 2026. Dólar por 1 milhão de tokens, preço de lista, sem cache e sem IOF:

| Papel neste plano | Modelo | Entrada | Saída |
|---|---|---|---|
| Modelo fraco | `gpt-4o-mini` | US$ 0,15 | US$ 0,60 |
| Modelo forte | `gpt-5` | US$ 1,25 | US$ 10,00 |

Cotação usada: **1 USD = R$ 5,57** (comercial em 10 nov 2026). R$ 20 equivalem a US$ 3,59. A escolha de N, em todos os experimentos, usa `gpt-4o-mini`. É uma chamada curta e não justifica o modelo forte.

`gpt-4o`, `gpt-4.1` e `gpt-5.4` ficam de fora. No mesmo teto de contexto, uma tarefa com geradora e juíza nesses modelos passa de R$ 1,30 a vários reais, e a amostra forte cairia para poucas funções.

## Tokens reservados por tarefa

A reserva usa os tetos da seção Decisões de `docs/arquitetura.md`, com **5 voltas**. Uma volta é uma chamada da geradora e uma da juíza. A trajetória é uma só. A conta assume que as 5 voltas acontecem e que cada prompt chega no tamanho máximo. Parar antes gasta menos. A amostra não aumenta com a sobra.

Conversão usada: 4 caracteres por token.

| Bloco | Caracteres no teto | Tokens |
|---|---|---|
| Contexto de tipo | 12_000 | 3_000 |
| 16 trechos de código | 16_000 | 4_000 |
| Docstring + assinatura | 2_000 | 500 |
| Envelope do template | — | 200 |
| Contexto fixo de uma chamada | — | 7_700 |
| Uma versão completa, só na geradora a partir da 2ª volta | 6_000 | 1_500 |
| Recorte de uma versão na juíza | 1_500 | 375 |

Por tarefa, no teto:

| Chamada | Vezes | Entrada | Saída |
|---|---|---|---|
| Escolha de N (`gpt-4o-mini`) | 1 | 500 | 32 |
| Geradora, 1ª volta | 1 | 7_700 | 1_024 |
| Geradora, voltas 2 a 5 | 4 | 9_200 cada | 1_024 cada |
| Juíza, 5 voltas | 5 | 9_575 cada | 256 cada |

Soma da geradora: 44_500 de entrada e 5_120 de saída. Soma da juíza: 47_875 de entrada e 1_280 de saída. A juíza vê no máximo 5 versões, porque só existem 5 voltas.

Custo reservado de uma tarefa:

| Par | USD | R$ |
|---|---|---|
| Geradora fraca + juíza fraca | 0,0178 | 0,10 |
| Geradora fraca + juíza forte | 0,0825 | 0,46 |
| Geradora forte + juíza forte | 0,1804 | 1,00 |

Tentativas extras por falha de rede entram nessa mesma reserva. Se a próxima tentativa furaria a reserva da tarefa, a tarefa para com a última versão já produzida e não há nova chamada.

## Amostras

RustEval tem 90 tarefas. JavaEval tem 199 tarefas nos JSON do artigo (`results/java/catcoder/`, campo `fn_codes`). A amostra é um embaralhamento com `random.Random(0)` dos índices `0 .. N-1`. Os pares com modelo forte reutilizam o começo dessa lista, para comparar juíza fraca, juíza forte e par forte nas mesmas funções.

Rust, 46 índices:

`8, 7, 29, 44, 19, 11, 26, 75, 41, 57, 48, 70, 50, 84, 1, 34, 22, 6, 37, 43, 23, 47, 24, 79, 72, 83, 88, 52, 10, 55, 2, 54, 67, 82, 14, 77, 4, 81, 15, 21, 31, 85, 89, 59, 66, 63`

Java, 30 índices:

`92, 59, 173, 137, 58, 116, 138, 159, 88, 6, 157, 94, 100, 167, 68, 119, 166, 26, 99, 170, 22, 101, 148, 109, 121, 104, 32, 17, 152, 106`

O arquivo Arrow de Java não está neste checkout (só `dataset_info.json` e `state.json`). A rodada Java usa o dataset JavaEval de 199 tarefas, o mesmo cujas gerações estão em `results/java/`.

## Rodadas

Seis rodadas. Nenhuma repete In-File, RepoCoder, FULL, -CR ou -TC.

| Id | Língua | Geradora | Juíza | Índices | Tarefas | R$ reservados |
|---|---|---|---|---|---|---|
| R-FF | Rust | `gpt-4o-mini` | `gpt-4o-mini` | os 46 | 46 | 4,56 |
| R-JF | Rust | `gpt-4o-mini` | `gpt-5` | os 8 primeiros dos 46 | 8 | 3,68 |
| R-FFORTE | Rust | `gpt-5` | `gpt-5` | os 4 primeiros dos 46 | 4 | 4,02 |
| J-FF | Java | `gpt-4o-mini` | `gpt-4o-mini` | os 30 | 30 | 2,97 |
| J-JF | Java | `gpt-4o-mini` | `gpt-5` | os 6 primeiros dos 30 | 6 | 2,76 |
| J-FFORTE | Java | `gpt-5` | `gpt-5` | os 2 primeiros dos 30 | 2 | 2,01 |
| **Total** | | | | | **96 execuções de tarefa** | **19,99** |

Os 8 de Rust e os 6 de Java já estão dentro das listas maiores. O gasto conta de novo porque o ciclo roda outra vez, com outra juíza ou com as duas pontas fortes. Funções distintas: 46 em Rust (de 90) e 30 em Java (de 199).

A ordem de execução é a da tabela. Antes de cada tarefa, soma-se a reserva dela ao que já foi medido nas respostas da API. Se a soma passar de R$ 20, a tarefa não começa e as rodadas seguintes não começam.

O que cada par responde, dentro da contribuição:

- Fraca/fraca: o ciclo com o modelo barato, na amostra maior. É o ponto de comparação com o "salto maior em modelos menores" do slide.
- Fraca/forte: a mesma geradora, com a juíza forte, nas mesmas funções do começo da lista.
- Forte/forte: o teto do par forte, em poucas funções compartilhadas com as duas linhas anteriores.

## Métrica da contribuição

Para cada tarefa da amostra, uma trajetória. O código entregue é o da ação Parar, ou o da última versão se as 5 voltas se esgotarem. A tarefa conta como sucesso se esse código compila e passa em todos os testes unitários da tarefa (`TestAdapter.test` no Rust, Defects4J no Java).

A taxa da rodada é o número de sucessos dividido pelo número de tarefas dessa rodada. Na figura ela aparece com o tamanho da amostra e o par de modelos. Ela não entra na média do pass@1 do artigo. O pass@1 do artigo é o estimador sobre 10 amostras independentes do benchmark inteiro. Aqui há uma trajetória e um subconjunto fixo.

## O que é copiado do artigo

Ler `lastest_eval.pass@1` e não gerar de novo.

Generalização, série In-File, RepoCoder e Our Approach (CatCoder):

| Série do slide | Pasta |
|---|---|
| In-File | `results/{java,rust}/infile/` |
| RepoCoder | `results/{java,rust}/repocoder/` |
| Our Approach | `results/{java,rust}/catcoder/` |

Ablação:

| Série do slide | Pasta |
|---|---|
| FULL | `results/{java,rust}/catcoder/` |
| -CR | `results/{java,rust}/-cr/` |
| -TC | `results/{java,rust}/-tc/` |

Java, nas três séries da generalização, tem JSON para CodeGemma-7B, Llama-3-8B, CodeLlama-7B, CodeQwen-7B, DeepSeek-Coder-6.7B e CodeLlama-13B. A média do slide é a média desses seis pass@1, recalculada a partir dos JSON, sem nova inferência.

Rust, neste checkout, tem JSON só de CodeLlama-13B em `infile`, `repocoder`, `catcoder`, `-cr` e `-tc`. Java, em `-cr` e `-tc`, também só tem CodeLlama-13B. As outras barras de Rust e da ablação que aparecem no slide e não têm arquivo em `results/` são transcritas do artigo. Não são reexecutadas e não são lidas no olho a partir do screenshot.

## Figuras finais

Duas figuras, cada uma com o painel do artigo e um painel novo. Os modelos da OpenAI não ocupam o eixo de CodeLlama ou CodeQwen.

**Generalização.** Painel do artigo: as barras In-File, RepoCoder e Our Approach, copiadas, em Java e em Rust, como nos gráficos (a) e (b) do slide. Painel da contribuição: a taxa de sucesso na amostra, em Java e em Rust, para os três pares (fraca/fraca, fraca/forte, forte/forte), com o n de cada barra na legenda (30, 6 e 2 em Java; 46, 8 e 4 em Rust).

**Ablação.** Painel do artigo: FULL, -CR e -TC, copiados, como nos gráficos (c) e (d). Painel da contribuição: uma série só, o sistema completo (busca híbrida + grau N + ciclo), nas mesmas seis barras do painel novo da generalização. Não há amostra de -CR nem de -TC da contribuição. Cada uma dessas variantes repetiria as 96 execuções.

## Registro de cada tarefa

Por tarefa, guardar: índice, língua, id da rodada, modelo da geradora, modelo da juíza, N devolvido e N usado depois do corte em 4, número de voltas, ação final, código entregue, se compilou, se passou nos testes, tokens de entrada e de saída medidos na API e o custo em dólar convertido a R$ 5,57. O acumulado em reais fica no mesmo registro, para o corte em R$ 20 ser verificável.
