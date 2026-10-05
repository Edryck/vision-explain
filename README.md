# IC4 — Pequenos Modelos Vision-Language para Explicação de Resultados de Visão Computacional

Aluno: Edryck · Projeto de IC (12 meses / 52 semanas)

Pipeline central: IMAGEM → modelo de CV → evidências estruturadas → Small VLM (SmolVLM) → explicação.

## Estrutura

```
index.html     painel de acompanhamento da IC (deploy no Vercel)
data/          datasets e subconjuntos de avaliação (dados pesados no .gitignore)
notebooks/     notebooks de exploração (00_setup.ipynb = Semana 1)
src/           código reutilizável (extração de evidências, prompts, métricas)
experiments/   registros e resultados dos experimentos
docs/          protocolos, rascunhos dos artigos, relatório final
```

## Ambiente (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
jupyter lab   # abrir notebooks/00_setup.ipynb
```

> Antes de escolher o Small VLM, confira a VRAM: `nvidia-smi` (se tiver GPU NVIDIA).
> Sem GPU, o SmolVLM-256M/500M roda em CPU, só que mais devagar.

## Painel

O `index.html` é estático (dados salvos no `localStorage` do navegador de quem abre).
Para o professor ver o painel, basta o link do Vercel.
