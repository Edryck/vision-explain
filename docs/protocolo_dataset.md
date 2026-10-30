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

## Limitações conhecidas

- Uma só categoria; resultados não se estendem a texturas ou objetos deformáveis.
- 10 imagens por grupo dão poder estatístico baixo; reportar tamanho de efeito e exemplos, não só médias.
- O mesmo PatchCore que gera a evidência erra (falso positivo ou negativo). A análise deve separar casos em que a evidência estava certa e errada.
