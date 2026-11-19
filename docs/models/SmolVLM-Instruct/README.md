# SmolVLM-Instruct (linha de base)

- Hugging Face: [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- Organização: Hugging Face
- Publicação no Hugging Face: 18/11/2024
- Parâmetros: ~2,2B
- Licença: Apache 2.0
- Tarefa: `image-text-to-text` (VLM)
- Artigo: [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299)

## Arquitetura

- Encoder SigLIP + LM SmolLM2 1.7B, ligados por projeção com *pixel shuffle*, que reduz os tokens de imagem.
- A imagem é dividida em *tiles* de 384 px, mais uma versão reduzida da imagem inteira.
- Sem pós-treino com RL.

## Carregamento no projeto

- Classe: `AutoModelForImageTextToText`. Ambiente `latest` do benchmark.
- `longest_edge=1152`, definido no exp003. O padrão de 1536 px ampliava as fotos do MVTec (900 px) e gastava tokens sem ganhar detalhe.
- É o modelo de `src/vlm.py`, usado nos exp003 a exp007 (Configurações A e B) e como juiz do FaithScore simplificado.
- Registro em `src/benchmark.py` (`MODELS["SmolVLM-Instruct"]`, `BASELINE`).

## Uso no benchmark (Configuração A)

- Entrada: imagem + `Question: <pergunta>` (`src.prompts.build_prompt_a`), sem evidência.
- Geração gulosa (`do_sample=False`), até 128 tokens novos.
- Entra em `bench_bottle.ipynb` e `bench_multiclass.ipynb`. Também roda nos notebooks de inferência quando `RUN_BASELINE = True`.
- É a referência do McNemar: cada modelo novo é comparado com ele nos mesmos pares (imagem, pergunta).
- Saída: `experiments/bench/<subconjunto>/SmolVLM-Instruct.json`.

## Diferença em relação ao exp003 e ao exp007

- O exp003 e o exp007 usaram *KV cache* do prefixo da imagem (`generate_answers_shared_image`). O benchmark gera cada pergunta do zero para tratar todos os modelos igual.
- As respostas devem coincidir com as do exp003 na mesma GPU e no mesmo dtype. Diferença pequena entre CPU fp32 e GPU fp16 é esperada.
- Com o juiz do FaithScore sendo o próprio SmolVLM, a métrica pode favorecer esta família. Por isso o FaithScore é secundário no benchmark.
