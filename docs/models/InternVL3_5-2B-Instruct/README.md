# InternVL3_5-2B-Instruct

- Hugging Face: [OpenGVLab/InternVL3_5-2B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-2B-Instruct)
- Organização: Shanghai AI Lab (OpenGVLab)
- Publicação no Hugging Face: 25/08/2025
- Parâmetros: ~2,3B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- Mesma arquitetura do `InternVL3_5-2B` (InternViT-300M + Qwen3, *tiles* dinâmicos de 448 px).
- Checkpoint intermediário: só CPT + SFT, sem o Cascade RL. A OpenGVLab publica os estágios para pesquisa.

## Carregamento no projeto

- Igual ao `InternVL3_5-2B`: código remoto, ambiente `legacy` (transformers 4.57.1), `use_flash_attn=False`, até 12 *tiles*.
- Registro em `src/benchmark.py` (`MODELS["InternVL3_5-2B-Instruct"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/InternVL3_5-2B-Instruct.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/InternVL3_5-2B-Instruct.json`.

## Por que entra no benchmark

- Não é uma versão "melhor" do 2B: é o estágio anterior ao RL. Serve como controle do efeito do pós-treino com RL.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
