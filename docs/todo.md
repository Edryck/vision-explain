# Guia da Semana 6: extração de evidências estruturadas

Objetivo: escrever `src/extract_evidence.py`, uma função que recebe uma imagem, roda o PatchCore (via `anomalib`) e devolve um JSON com as evidências. Rodar em 5 imagens e salvar em `experiments/exp002_evidencias.json`.

Pipeline desta semana: imagem → PatchCore → mapa de anomalia → regiões → JSON → (esboço do prompt do VLM).

Os nomes de classes e argumentos do `anomalib` mudam entre versões. Antes de copiar qualquer exemplo, rode `pip show anomalib` e leia a documentação da versão instalada.

## 1. Preparação

- [ ] Instalar `anomalib` no `.venv` e adicionar ao `requirements.txt`.
- [ ] Conferir `python -c "import anomalib; print(anomalib.__version__)"`.
- [ ] Baixar o MVTec AD pelo site oficial e extrair em `data/mvtec_ad/` (a pasta `data/` já é ignorada pelo git).
- [ ] Escolher 1 categoria para começar (sugestão: `bottle`, `hazelnut` ou `screw`). Uma categoria basta nesta semana.
- [ ] Olhar a estrutura da pasta: `train/good`, `test/<tipo_de_defeito>`, `ground_truth/<tipo_de_defeito>`. Entender o que o modelo vê no treino e o que vê no teste.

## 2. Entender o PatchCore antes de codar

- [ ] Ler o resumo e a seção de método do artigo do PatchCore.
- [ ] Responder por escrito, com suas palavras:
  - O que é o "memory bank" e do que ele é feito?
  - Por que não há treino por gradiente?
  - Como o score de anomalia de uma imagem é calculado a partir do mapa?
  - O que o coreset subsampling faz e por que existe?
- [ ] Perguntas que decidem o formato do JSON:
  - Qual a resolução do mapa de anomalia em relação à imagem original?
  - O score é comparável entre categorias? (Em geral não: cada categoria tem sua escala.)
  - O modelo diz o tipo de defeito? (Não. Só localiza e pontua.)

## 3. Rodar o modelo em uma imagem

- [ ] Treinar (ajustar) o PatchCore na categoria escolhida com `Engine` do `anomalib`.
- [ ] Rodar a predição em 1 imagem de teste com defeito e 1 sem defeito.
- [ ] Inspecionar a saída: quais campos existem (score, rótulo previsto, mapa, máscara)? Qual o `shape` e o `dtype` de cada um?
- [ ] Salvar uma figura com a imagem, o mapa sobreposto e a máscara de ground truth, para você enxergar o que o modelo enxerga.

## 4. Converter a saída em regiões

O mapa de anomalia é uma matriz de números. O VLM precisa de regiões descritas em texto. Passos:

- [ ] Normalizar o mapa e escolher um limiar (threshold). O `anomalib` calcula um limiar adaptativo no ajuste; verifique como acessá-lo na sua versão.
- [ ] Binarizar o mapa com esse limiar.
- [ ] Separar componentes conectados: cada componente vira uma região candidata.
- [ ] Para cada região, calcular: caixa delimitadora (bbox), área em pixels, fração da imagem ocupada, centroide, score máximo e médio dentro da região.
- [ ] Traduzir posição numérica para linguagem: por exemplo, dividir a imagem em uma grade 3x3 e nomear a célula ("canto superior esquerdo", "centro", etc.).
- [ ] Decidir o que fazer quando não há região acima do limiar (imagem normal) e quando há muitas regiões minúsculas (ruído). Sugestão: descartar regiões abaixo de uma área mínima e limitar o número de regiões por imagem.

Documentação útil: `cv2.connectedComponentsWithStats` (OpenCV) ou `skimage.measure.label` e `skimage.measure.regionprops` (scikit-image).

## 5. Desenhar o JSON

Rascunho de esquema (ajuste conforme o que você descobrir na etapa 3):

```json
{
  "image": "data/mvtec_ad/bottle/test/broken_large/000.png",
  "category": "bottle",
  "model": {"name": "PatchCore", "backbone": "wide_resnet50_2", "library": "anomalib"},
  "image_size": {"width": 900, "height": 900},
  "image_score": 0.87,
  "threshold": 0.50,
  "is_anomalous": true,
  "regions": [
    {
      "id": 1,
      "bbox": [x_min, y_min, x_max, y_max],
      "area_px": 12345,
      "area_fraction": 0.015,
      "centroid": [cx, cy],
      "position": "centro-esquerda",
      "max_score": 0.91,
      "mean_score": 0.72
    }
  ]
}
```

Perguntas de projeto para decidir e registrar:

- Coordenadas em pixels absolutos ou normalizadas entre 0 e 1? (Normalizadas ajudam quando o VLM redimensiona a imagem.)
- O score vai cru ou normalizado? Como explicar ao VLM o que "0.87" significa?
- Incluir o nome da categoria ("bottle") no JSON? Isso é uma evidência ou uma dica que vaza informação?
- Qual o campo mínimo sem o qual o JSON deixa de ser útil?

## 6. Escrever `src/extract_evidence.py`

Regras do repositório: nomes em inglês, sem comentários no código.

- [ ] Uma função principal, por exemplo `extract_evidence(image_path, model, threshold=None) -> dict`.
- [ ] Funções pequenas separadas: carregar o modelo, rodar a predição, mapa → regiões, região → dicionário, posição → texto.
- [ ] A função de carregar o modelo não deve retreinar a cada chamada. Treine uma vez, salve o checkpoint e carregue.
- [ ] Validar o JSON antes de devolver: campos obrigatórios presentes, bbox dentro da imagem, números convertidos para tipos nativos do Python (`float`, `int`), não `numpy` nem `torch` (senão `json.dump` falha).
- [ ] Opcional: um bloco `if __name__ == "__main__"` ou um notebook `02_extract_evidence.ipynb` para chamar a função nas 5 imagens.

## 7. Testar em 5 imagens variadas

Escolha de propósito casos diferentes:

- [ ] 1 imagem normal (sem defeito).
- [ ] 2 imagens com tipos diferentes de defeito.
- [ ] 1 defeito pequeno.
- [ ] 1 defeito grande.
- [ ] Para cada uma, comparar as regiões extraídas com a máscara de ground truth (visualmente ou por IoU) e anotar acertos e erros.
- [ ] Salvar tudo em `experiments/exp002_evidencias.json` com: versão do `anomalib`, categoria, limiar, parâmetros e a lista de evidências.

Critério de conclusão da semana: a função roda nas 5 imagens e cada saída é um JSON válido. Confira com `json.loads(json.dumps(result))` e abra o arquivo final em um validador.

## 8. Esboço do template de prompt (começar já)

A dica da semana é pensar em como o JSON vira texto. Escreva em `docs/prompt_template_draft.md`:

- [ ] Uma versão do prompt Configuração A (só imagem + pergunta).
- [ ] Uma versão Configuração B (imagem + evidências em texto + pergunta).
- [ ] Uma função de conversão JSON → texto, em frases curtas. Exemplo de saída desejada:

```
The anomaly detector flagged the image as anomalous (score 0.87).
Region 1: center-left, covers 1.5% of the image, bounding box (x1, y1, x2, y2).
```

- [ ] Decidir o idioma do prompt (o SmolVLM funcionou em inglês no notebook 01).
- [ ] Anotar o risco: o VLM pode repetir o score como se fosse um fato visual, ou inventar o tipo de defeito. Isso vira hipótese para os experimentos futuros.

## 9. O que pesquisar

- Como o `anomalib` expõe `anomaly_map`, `pred_score`, `pred_mask` e o limiar na sua versão.
- Diferença entre score de imagem e mapa por pixel no PatchCore.
- Por que o limiar é por categoria e como o `anomalib` o escolhe.
- Como converter tensores (`torch`) em tipos nativos para JSON.
- Métricas padrão do MVTec AD: AUROC de imagem e de pixel (apenas para saber o que o modelo já atinge na literatura).

## 10. Documentação e leitura

- Documentação do `anomalib`: https://anomalib.readthedocs.io
- Repositório do `anomalib`: https://github.com/open-edge-platform/anomalib (confira se o endereço mudou)
- PatchCore, Roth et al. 2022, "Towards Total Recall in Industrial Anomaly Detection": https://arxiv.org/abs/2106.08265
- MVTec AD (download e descrição): https://www.mvtec.com/company/research/datasets/mvtec-ad
- Artigo MVTec AD (E6), seção do dataset e das métricas: https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html
- OpenCV, componentes conectados: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- scikit-image, `regionprops`: https://scikit-image.org/docs/stable/api/skimage.measure.html
- Módulo `json` do Python: https://docs.python.org/3/library/json.html

## 11. Armadilhas esperadas

- Rodar em CPU é lento. No notebook 01 o SmolVLM levou muito tempo por pergunta. O PatchCore em CPU também pode demorar na extração de features do treino; comece com 1 categoria e poucas imagens para testar o código, e rode a versão completa depois.
- O mapa de anomalia costuma ter resolução menor que a imagem original; reescale antes de calcular bbox, senão as coordenadas ficam erradas.
- Valores `numpy.float32` quebram `json.dump`.
- Limiar mal escolhido gera regiões demais (ruído) ou nenhuma. Registre o limiar usado no JSON.
- Não reajuste o limiar olhando as 5 imagens de teste para "dar certo". Fixe o critério antes e anote o que acontece.

## 12. Evidências e registros da semana

- `src/extract_evidence.py`
- `experiments/exp002_evidencias.json`
- Entrada em Experimentos (exp002): hipótese, dataset, modelo, parâmetros, resultado, análise.
- Entrada no Diário Científico com decisões (formato do JSON, limiar, tratamento de imagem normal).
- Observações da semana 6 no dashboard.
