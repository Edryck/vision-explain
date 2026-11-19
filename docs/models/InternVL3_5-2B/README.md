# InternVL3_5-2B

- Hugging Face: [OpenGVLab/InternVL3_5-2B](https://huggingface.co/OpenGVLab/InternVL3_5-2B)
- Organização: Shanghai AI Lab (OpenGVLab)
- Publicação no Hugging Face: 25/08/2025
- Parâmetros: ~2,3B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- Paradigma ViT–MLP–LLM: InternViT-300M + LM Qwen3.
- Alta resolução dinâmica: a imagem vira até 12 *tiles* de 448 px + miniatura; cada *tile* vira 256 tokens após *pixel shuffle*.
- Pipeline completo: CPT + SFT + Cascade RL (MPO offline + GSPO online). É a versão final da família.

## Carregamento no projeto

- Código remoto (`trust_remote_code=True`, `AutoModel` + `model.chat`). Quebra no transformers 5.x, então roda no ambiente `legacy` (venv com transformers 4.57.1, criado pelo notebook).
- `use_flash_attn=False` (sem flash-attn no Kaggle).
- Pré-processamento do card reimplementado em `src.benchmark.internvl_tiles` (máximo de 12 *tiles*). Uma foto 900x900 vira 2x2 *tiles* + miniatura = 5 *tiles*.
- `chat()` não devolve o tamanho do prompt: `prompt_tokens` fica `None` e o custo de imagem é registrado em `n_tiles`.
- Existe a versão `OpenGVLab/InternVL3_5-2B-HF` no formato nativo do transformers, mas ela não tem par `-Instruct`. Para os dois InternVL rodarem igual, ambos usam o código remoto.
- Registro em `src/benchmark.py` (`MODELS["InternVL3_5-2B"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/InternVL3_5-2B.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/InternVL3_5-2B.json`.

## Por que entra no benchmark

- Comparar com `InternVL3_5-2B-Instruct` isola o efeito do Cascade RL: mesmos pesos de partida, mesma arquitetura.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
