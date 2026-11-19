# Qwen3-VL-4B-Instruct

- Hugging Face: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- Organização: Alibaba (Qwen)
- Publicação no Hugging Face: 11/10/2025
- Parâmetros: ~4,4B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- Mesma arquitetura do Qwen3-VL-2B (Interleaved-MRoPE, DeepStack, resolução nativa), com LM maior.
- Edição *Instruct* (sem raciocínio).

## Carregamento no projeto

- Classe: `AutoModelForImageTextToText` (`Qwen3VLForConditionalGeneration`). Ambiente `latest`.
- Em fp16 ocupa cerca de 9 GB de pesos; cabe numa T4 de 16 GB com uma imagem por vez.
- Registro em `src/benchmark.py` (`MODELS["Qwen3-VL-4B-Instruct"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/Qwen3-VL-4B-Instruct.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/Qwen3-VL-4B-Instruct.json`.

## Por que entra no benchmark

- Maior modelo do benchmark; serve de teto dentro do limite de "small" (até ~4B).
- Comparar com o 2B da mesma família isola o efeito de tamanho.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
