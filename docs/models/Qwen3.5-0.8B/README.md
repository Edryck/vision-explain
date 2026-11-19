# Qwen3.5-0.8B

- Hugging Face: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- Organização: Alibaba (Qwen)
- Publicação no Hugging Face: 28/02/2026
- Parâmetros: ~0,87B (LM de 0,8B + encoder de visão)
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)

## Arquitetura

- Modelo multimodal nativo (*early fusion*): LM causal com encoder de visão, treinado com tokens multimodais desde o pré-treino.
- LM híbrido de 24 camadas: blocos de Gated DeltaNet (atenção linear) intercalados com Gated Attention (6 × (3 × DeltaNet → 1 × Attention)).
- Processador de imagem do Qwen3-VL (`Qwen3VLProcessor`): patch 16, *merge* 2x2, resolução nativa (sem resize fixo).
- Pós-treino com RL em larga escala. Suporta modos com e sem raciocínio (*thinking*).

## Carregamento no projeto

- Classe: `AutoModelForImageTextToText` (`Qwen3_5ForConditionalGeneration`).
- Precisa de transformers recente (5.x); não existe no 4.57. Roda no ambiente `latest`.
- O 0.8B já vem sem *thinking* por padrão; mesmo assim o benchmark passa `enable_thinking=False` no `apply_chat_template` e descarta qualquer bloco `<think>` que sobrar.
- Os parâmetros de amostragem sugeridos no card (temperature 0,7, top_p 0,8) são ignorados: o benchmark usa geração gulosa em todos os modelos.
- Registro em `src/benchmark.py` (`MODELS["Qwen3.5-0.8B"]`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos (`src.benchmark.MAX_NEW_TOKENS`).
- Subconjuntos: `bottle` (`data/subset_avaliacao.json`, 120 respostas) e `multiclass` (subconjunto do exp007, 2220 respostas).
- Métrica principal: acerto por gabarito (`src.multicategory.score_response`). FaithScore opcional, com juiz fixo `SmolVLM-Instruct`.
- Notebooks: `experiments/notebooks/inferences/Qwen3.5-0.8B.ipynb`, `experiments/notebooks/bench/bench_bottle.ipynb` e `bench_multiclass.ipynb`.
- Saída: `experiments/bench/<subconjunto>/Qwen3.5-0.8B.json`.

## Por que entra no benchmark

- Menor modelo do benchmark e o único lançado em 2026. Bom para medir o piso de tamanho.
- O card avisa que o 0.8B entra em laço de raciocínio com mais facilidade que os maiores no modo *thinking*; por isso ele fica desligado.

## Pendências

- Não validado ainda: carregamento, tempo por resposta e memória no Kaggle.
