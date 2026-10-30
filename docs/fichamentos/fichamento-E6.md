# Fichamento E6

Título: MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection (Bergmann, Fauser, Sattlegger, Steger, 2019, CVPR 2019)

## Problema

Detecção não supervisionada de anomalias é central na inspeção industrial: defeitos são raros e de tipo desconhecido, então só há amostras normais para treino. Faltava um dataset de imagens naturais que fosse ao mesmo tempo abrangente (vários objetos e texturas), com vários tipos de defeito e com anotação de região em nível de pixel. Os datasets existentes eram de classificação adaptada (MNIST, CIFAR10, ImageNet com classes tratadas como outliers, em que o "anômalo" difere muito da distribuição de treino), de uma só textura (NanoTWICE, 45 imagens) ou sintéticos (DAGM, com anotação grosseira em elipses).

## Método

O artigo é de dataset, não propõe um método novo. Contribuições:

1. O dataset MVTec AD: 5 texturas e 10 objetos, 5354 imagens coloridas de alta resolução (700x700 a 1024x1024), capturadas com sensor industrial de 2048x2048 e lentes telecêntricas, sob iluminação controlada.
2. Um benchmark inicial com 6 métodos, avaliados em classificação de imagens anômalas e em segmentação de regiões anômalas.
3. Um protocolo para definir o limiar de segmentação usando apenas imagens normais de validação: define-se uma área mínima de defeito e aumenta-se o limiar até a maior região anômala da validação ficar abaixo dessa área.

Métodos do benchmark: AnoGAN, autoencoders convolucionais (perda L2 e SSIM), dicionário de features de CNN (ResNet-18 + PCA + k-means), GMM para textura (HALCON) e modelo de variação (média e desvio por pixel, exige alinhamento).

## Dataset

- 15 categorias. Texturas: carpet, grid, leather, tile, wood. Objetos: bottle, cable, capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper.
- Treino: 3629 imagens, todas sem defeito. Teste: 1725 imagens, sendo 467 sem defeito e 1258 com defeito.
- 73 tipos de defeito (média de 5 por categoria): arranhões, amassados, contaminação, defeitos estruturais (peça deformada) e ausência de partes.
- 1888 regiões de defeito anotadas manualmente com máscara de ground truth em nível de pixel.
- Bottle (a categoria que uso): 209 imagens de treino, 20 de teste sem defeito, 63 de teste com defeito, 3 grupos de defeito (broken_large, broken_small, contamination), 68 regiões, imagens de 900x900.
- Parte dos objetos está alinhada (toothbrush, capsule, pill), parte tem rotação aleatória (metal nut, screw, hazelnut). Grid, screw e zipper são em escala de cinza.

## Baseline

Seis métodos: AnoGAN, AE (SSIM), AE (L2), CNN Feature Dictionary, Texture Inspection (GMM) e Variation Model. Limiares estimados só com imagens de validação sem defeito. Aumento de dados (translação, rotação, espelhamento) para chegar a 10000 patches por categoria.

## Métricas

- Classificação: proporção de imagens corretamente classificadas, separada em sem defeito e com defeito (Tabela 2).
- Segmentação: sobreposição relativa por região (per-region overlap) com o ground truth e ROC AUC, que independe do limiar (Tabela 3).
- Taxa de verdadeiros positivos medida em pixels e taxa de falsos positivos em pixels.

## Resultados

- Nenhum método foi bom em todas as categorias. Nas texturas, o dicionário de features de CNN foi o mais consistente em ROC AUC. Nos objetos, os autoencoders foram os melhores, com o L2 melhor em sobreposição por região.
- Bottle: AE (SSIM) classificou 0,85 das imagens normais e 0,90 das anômalas. A sobreposição por região foi baixa (máximo 0,22 com AE L2) e o ROC AUC chegou a 0,93 com AE SSIM.
- ROC AUC alto não garante boa segmentação com o limiar estimado: o mapa de anomalia pode representar bem o defeito e o limiar estimado só com imagens normais ainda assim errar.
- Autoencoders falham em detalhes finos e texturas de alta frequência (blur, falsos positivos em tile e zipper). AnoGAN sofre colapso de modo. O modelo de variação falha quando há variação natural (metal nut, capsule).
- Os próprios autores dizem que há "considerável espaço para melhora".

## Limitações

- Só imagens de inspeção industrial, em ambiente controlado (iluminação e fundo). Não generaliza para cenas naturais abertas.
- Cada categoria é treinada e avaliada separadamente; os métodos não são comparáveis entre categorias e os scores têm escala própria.
- O benchmark é de 2019 e não inclui métodos baseados em memory bank (PatchCore, de 2022, é posterior).
- Anotação só localiza o defeito. O tipo do defeito existe só como nome da pasta (ex.: broken_large), não como rótulo de classe no benchmark.
- Estimar o limiar sem imagens anômalas é difícil, como o artigo mostra.
- Poucas imagens de teste em algumas categorias (toothbrush tem 12 normais e 30 anômalas).

## Ideias reutilizáveis

1. Usar o MVTec AD como domínio do projeto: tem máscara de ground truth, o que permite verificar se a evidência do PatchCore e a resposta do VLM batem com a região real.
2. Estrutura de pastas `train/good`, `test/<tipo>`, `ground_truth/<tipo>` para montar o `data/subset_avaliacao.json`.
3. Nome do tipo de defeito (pasta de teste) como gabarito para a pergunta "qual o tipo de defeito?", embora o modelo de CV não o forneça.
4. Métricas de classificação separadas em normal e anômalo, e ROC AUC independente de limiar, para descrever a qualidade do PatchCore antes de usá-lo como evidência.
5. Protocolo do limiar com validação só de imagens normais, útil para justificar o limiar da extração de evidências.
6. Categorias com e sem rotação, e em escala de cinza, como fator de dificuldade a controlar ao escolher o subconjunto.

## Relação com o projeto

É o dataset do pipeline imagem → CV → evidências → Small VLM. O PatchCore gera o mapa de anomalia e as regiões estruturadas que entram no prompt da Configuração B; o MVTec AD fornece imagens normais para ajustar o PatchCore e imagens com defeito mais máscara para checar a evidência. As perguntas padronizadas do subconjunto de avaliação ("há defeito? onde? qual o tipo?") se apoiam nos nomes de defeito das pastas de teste.

Riscos: a categoria `bottle` é simples (objeto rígido, alinhado, 3 tipos de defeito), então o ganho da Configuração B pode não generalizar para outras categorias. O PatchCore localiza e pontua, mas não nomeia o defeito, então a pergunta sobre o tipo depende do VLM ou do gabarito. As imagens são de inspeção industrial, um domínio em que o SmolVLM provavelmente viu pouco no treino.
