# Protocolo do subconjunto de avaliação

Arquivo de configuração: `data/subset_avaliacao.json`. Gerado por `src/build_subset.py` (semente 42, reprodutível).

## Dataset e categoria

- MVTec AD (Bergmann et al., 2019), categoria `bottle`.
- Motivo da escolha: objeto rígido, alinhado, imagens de 900x900 e só 3 tipos de defeito, o que simplifica o primeiro ciclo de experimentos. A limitação é a generalização para outras categorias.
- Todas as imagens vêm de `test/`. O PatchCore é ajustado apenas com `train/good` (209 imagens), então não há vazamento entre ajuste e avaliação.

## Composição (40 imagens)

| Grupo | Rótulo | Imagens |
|---|---|---|
| `good` | normal | 10 |
| `broken_large` | anômala | 10 |
| `broken_small` | anômala | 10 |
| `contamination` | anômala | 10 |

Balanceamento de 10 por grupo, amostrado sem reposição com semente fixa. Cada entrada traz `id`, `path`, `label`, `defect_type` e `mask_path` (máscara de ground truth, `null` nas normais).

## Perguntas padronizadas

As mesmas 3 perguntas valem para todas as imagens, nas Configurações A e B:

1. `q1_presence`: "Is there any defect in this image?"
2. `q2_location`: "Where is the defect located in the image?"
3. `q3_type`: "What type of defect is it?"

Total: 120 pares imagem-pergunta por configuração.

## Gabarito

- Presença: `label` (normal ou anômala).
- Localização: máscara em `mask_path`. A posição por célula de grade 3x3 pode ser derivada da máscara.
- Tipo: `defect_type` (nome da pasta). O PatchCore não fornece o tipo; ele só localiza e pontua.
- Nas imagens normais, as perguntas q2 e q3 não têm resposta correta além de "não há defeito". Respostas que inventam defeito contam como alucinação.

## Regras de comparação A vs B

- Mesmo modelo (SmolVLM-Instruct), mesma imagem, mesma pergunta, decodificação gulosa (`do_sample=False`) e mesmo `max_new_tokens`.
- O bloco `Question: ...` é idêntico. A única diferença é o bloco de evidências que a Configuração B adiciona antes dele (`src/prompts.py`).
- As evidências vêm de `src/extract_evidence.py` (PatchCore, limiar normalizado 0,5).

## Otimizações de inferência (semana 9)

A primeira execução em CPU levou cerca de 204 s por pergunta (cerca de 6,8 h para 120 perguntas). Ela foi interrompida com 7 respostas, descartadas. Toda a Configuração A foi refeita com as duas mudanças abaixo, que valem igualmente para a Configuração B.

- Resolução: `longest_edge = 1152` no processador (`LONGEST_EDGE` em `src/vlm.py`), no lugar do padrão 1536. As imagens do `bottle` têm 900x900; o padrão as ampliava. O valor 1152 equivale a 3 x 384 e reduz o número de tiles e de tokens visuais.
- Cache: as 3 perguntas de uma imagem compartilham o prefixo com a imagem. O prefixo é calculado uma vez e seu KV cache é copiado para cada pergunta (`generate_answers_shared_image`). Na Configuração B, o bloco de evidências é o mesmo nas 3 perguntas e também entra no prefixo.

Regras:

- As respostas de A e de B usam o mesmo `longest_edge` e o mesmo caminho de geração.
- A medição de custo (`src/measure_cost.py`) não usa o cache: mede uma pergunta isolada, do início ao fim, com a máquina ociosa.
- O `exp001` e o notebook 02 usaram `longest_edge = 1536`; não são diretamente comparáveis às respostas do `exp003`.
- O verificador da métrica usa imagem sem splitting (`do_image_splitting = False`), por custo.

## Métricas (semanas 10 e 11)

- Acerto por gabarito (`src/ground_truth_score.py`): compara a resposta com rótulo, máscara em grade 3x3 e tipo de defeito, por palavras-chave. Medida principal proposta, pendente de aprovação do orientador.
- FaithScore simplificado (`src/faithscore.py`): decomposição por regras e SmolVLM como verificador. Medida secundária; ver `docs/relatorio_fase3.md`.

## Limitações conhecidas

- Uma só categoria; resultados não se estendem a texturas ou objetos deformáveis.
- 10 imagens por grupo dão poder estatístico baixo; reportar tamanho de efeito e exemplos, não só médias.
- O mesmo PatchCore que gera a evidência erra (falso positivo ou negativo). A análise deve separar casos em que a evidência estava certa e errada.
