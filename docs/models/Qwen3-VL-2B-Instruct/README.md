# Qwen3-VL-2B-Instruct

- Hugging Face: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- Organização: Alibaba (Qwen)
- Publicação no Hugging Face: 19/10/2025
- Parâmetros: ~2,1B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- ViT + LM Qwen3 denso, edição *Instruct* (sem raciocínio; existe a irmã *Thinking*).
- Interleaved-MRoPE (posição em tempo, largura e altura) e DeepStack (funde várias camadas do ViT para detalhe fino).
- Resolução nativa dinâmica: o número de tokens de imagem depende do tamanho da foto.

## Carregamento no projeto

- Classe: `AutoModelForImageTextToText` (`Qwen3VLForConditionalGeneration`). Ambiente `latest`.
- Sem `flash_attention_2` (não vem instalado no Kaggle); usa a atenção padrão.
- Resolução padrão do processador; nenhuma redução é aplicada.
- Registro em `src/benchmark.py` (`MODELS["Qwen3-VL-2B-Instruct"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/Qwen3-VL-2B-Instruct.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/Qwen3-VL-2B-Instruct.json`.

## Por que entra no benchmark

- Mesma família do Qwen3-VL-4B: o par mede o efeito de escala dentro da família.
- Pós-treino com RL, ao contrário do SmolVLM.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
