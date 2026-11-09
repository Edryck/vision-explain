# exp007: especificações para o Artigo 1

Experimento intermediário aprovado pelo orientador: repetir A vs. B nas 15 categorias do MVTec AD e medir mais coisas além do acerto por gabarito. Este texto entra na Metodologia (seção de experimentos) quando o `artigo1.md` for criado na semana 19. Os números de resultado só entram depois da execução no Colab.

## Dados

- MVTec AD, 15 categorias, conjunto de teste.
- 50 imagens por categoria: 10 `good` e 40 anômalas. O `toothbrush` tem só 30 anômalas, então fica com 40 imagens. Total: 740 imagens, 2.220 respostas por configuração.
- Anômalas divididas igualmente entre os tipos de defeito da categoria (round-robin); quando um tipo acaba, os outros completam. Semente 42.
- Lista fixa em `experiments/exp007_multicategoria/subset.json`, gerada por `python -m src.multicategory`.
- O bottle é sorteado de novo por esse critério, então não é o mesmo subconjunto do exp003/exp005.

## Pipeline

- CV: PatchCore (anomalib 2.x, backbone `wide_resnet50_2`), um modelo por categoria, coreset 0,01 (igual ao bottle), limiar adaptativo normalizado em 0,5. Regiões: componentes conexos acima do limiar, mínimo 0,1% da imagem, até 5, posição na grade 3x3.
- VLM: `HuggingFaceTB/SmolVLM-Instruct`, decodificação gulosa, `max_new_tokens=128`, `longest_edge=1152`, cache do prefixo da imagem, batch 1.
- Prompts A e B iguais aos do exp003/exp005 (`src/prompts.py`). Perguntas q1 presença, q2 localização, q3 tipo.
- Hardware: Colab com GPU. bf16 em GPU com suporte (L4, A100); fp16 na T4. O tipo de GPU fica salvo no JSON de respostas (`environment.gpu`). Latência não é comparável com o notebook da Fase 4.

## Métricas

1. Acerto por gabarito (principal). q1 e q2 iguais ao bottle. q3 usa um dicionário de termos por tipo de defeito (73 tipos, `TYPE_TERMS` em `src/multicategory.py`), escrito antes de ver as respostas e permissivo de propósito. `combined` aceita os termos de qualquer tipo da categoria.
2. FaithScore simplificado (secundária, mesmo verificador do exp004).
3. Qualidade da evidência (PatchCore): AUROC de imagem e de pixel no teste completo da categoria; no subconjunto, acurácia de detecção, falsos positivos e negativos, IoU das caixas com a máscara, acerto de célula 3x3 e evidência correta (detecção certa e caixa tocando a máscara).
4. Fidelidade à evidência: q1 concorda com o veredito do PatchCore; q2 cita uma célula que a caixa do PatchCore toca; q2 copia literalmente a posição do relatório; q3 responde rótulo genérico ("anomaly"). Calculada para A também, como referência.
5. Consistência entre perguntas: proporção de imagens em que q1 afirma defeito e q2/q3 negam, ou o contrário.
6. Riqueza: tokens gerados, palavras, número de afirmações (decomposição do FaithScore) e taxa de respostas sem afirmação.

## Análise estatística

- McNemar exato pareado (A vs. B) no geral, por pergunta, por categoria e por evidência certa ou errada.
- Bootstrap de 2.000 reamostras por imagem (não por resposta) para o IC95% da diferença B − A, porque as 3 perguntas da mesma imagem não são independentes.

## O que este experimento resolve da Fase 4

- No bottle o PatchCore acertou tudo; em outras categorias ele erra, então dá para medir a B com evidência errada.
- n sobe de 40 para 740 imagens.

## Limitações já conhecidas

- O dicionário de q3 é heurístico e pode ser permissivo em tipos como `defective` e `rough`.
- Termos verticais de q2 (`neck`, `cap`) foram pensados para o bottle.
- FaithScore continua com o viés para "yes" visto no exp004.

## Como reproduzir

Notebook `notebooks/03_exp007_multicategoria_colab.ipynb`, em ordem: evidências (`src.configB.exp007_multicategoria.evidences`), A (`src.configA.exp007_multicategoria.run`), B (`src.configB.exp007_multicategoria.run`), métricas (`src.multicategory_metrics score|report`).
