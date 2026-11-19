# SmolVLM2-2.2B-Instruct

- Hugging Face: [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- Organização: Hugging Face
- Publicação no Hugging Face: 08/02/2025
- Parâmetros: ~2,2B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- Sucessor direto do `SmolVLM-Instruct` (linha de base do projeto), que é o `base_model` dele.
- Mesmo desenho: encoder SigLIP + LM SmolLM2 1.7B, imagem dividida em *tiles* de 384 px com *pixel shuffle*.
- Treinado também com vídeo e múltiplas imagens. Sem pós-treino com RL.

## Carregamento no projeto

- Classe: `AutoModelForImageTextToText` (`SmolVLMForConditionalGeneration`). Ambiente `latest`; o card pede `num2words`.
- `longest_edge=1152`, o mesmo ajuste do exp003 para a linha de base. Sem isso o processador amplia as fotos do MVTec para 1536 px.
- Registro em `src/benchmark.py` (`MODELS["SmolVLM2-2.2B-Instruct"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/SmolVLM2-2.2B-Instruct.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/SmolVLM2-2.2B-Instruct.json`.

## Por que entra no benchmark

- O par SmolVLM → SmolVLM2 mostra quanto a evolução da família muda o resultado, com tamanho e pré-processamento iguais.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
