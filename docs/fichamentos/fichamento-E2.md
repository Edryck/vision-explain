# Fichamento E2

Título: Small Vision-Language Models: A Survey on Compact Architectures and Techniques (Patnaik et al., 2025, arXiv:2503.10665)

## Problema
Os modelos multimodais de grande porte (como LLaVA-13B, GPT-4V e Flamingo) exigem recursos massivos de memória e processamento. Essa exigência deixa inviável a implantação em ambientes com recursos limitados, como smartphones, dispositivos móveis e sistemas embarcados.
O problema central da área é como reduzir significativamente a pegada de memória e a latência de inferência sem comprometer drasticamente a precisão e a capacidade de raciocínio multimodal.

## Método

Como é um survey, o artigo organiza e categoriza a literatura em uma taxonomia baseada em três paradigmas de arquitetura de backbone, baseados em Transformer, Mamba e Híbridos:

1. Modelos baseados em Transformer: Usam mecanismo de atenção tradicional, adaptando LLMs menores e usando técnicas de compressão e pré-fusão, por exemplo: TinyGPT-V, MiniGPT-4, LLaVA-Mini, Moondream2, Mini-Gemini e DeepSeek-VL2.
2. Modelos baseados em Mamba (State Space Models - SSMs): Trocam a atenção quadrática por modelos de espaço de estados com complexidade e escalabilidade linear em relação ao comprimento da sequência, por exemplo: VL-Mamba, Cobra, SiMBA, FusionMamba, SpectralMamba, 2DMamba e ML-Mamba.
3. Modelos híbridos: Combina camadas de Transformer, SSMs/Mamba, CNNs e modelos de difusão/máscara para equilibrar o processamento de detalhes visuais e contexto longo, por exemplo: FastVLM (CNN + Transformer), SAMBA, Zamba, Diffusion Mamba (DiM) e MaskMamba.

O survey também mapeia técnicas de eficiência como destilação de conhecimento (knowledge distillation), mecanismos de atenção leves e pré-fusão de modalidades (modality pre-fusion).

## Datasets de Instrução Mapeados

O artigo analisa como os sVLMs são treinados em duas etapas (alinhamento e ajuste fino por instrução), destacando diversos conjuntos de dados da literatura:
- Dados sintéticos e curados por instrução: LLaVA-Instruct-158K, o conjunto de 3500 pares detalhados do MiniGPT-4 e os 2,7 milhões de instruções/legendas do Mini-Gemini.
- Dados multimodais de pré-treino e diálogo: LAION-5B, Conceptual Captions (CC3M/CC12M), SBU, Multimodal C4 (M3W), DataCompDR-1B e SlimPajama.

## Baselines

O survey tabula a avaliação dos sVLMs comparando-os com VLMs maiores de referência (como LLaVA-1.5 de 7B/13B, Flamingo e InstructBLIP) nos seguintes benchmarks:
- Visual Question Answering (VQA): VQAv2, GQA, TextVQA, VizMiz, DocVQA, ChartQA.
- Raciocínio Geral e Multimodal: MMBench, ScienceQA, MM-Vet, MME, POPE, SEED-Bench, MMMU.
- Domínios específicos: Imagenet (classificação), ADE20K/COCO (segmentação e detecção), EgoSchema/MVBench (entendimento de vídeo) e PANDA/TCGA-BRCA (análise médica).

## Métricas

O artigo dividiu as métricas em duas categorias, sendo elas: Qualidade Multimodal e Eficiência Computacional.

- Métricas de Qualidade Multimodal:
  - Acurácia (%) e Acurácia Top-1: Para VQA, classificação e raciocínio de múltipla escolha.
  - Pontuação MME / MM-Vet / MMBench: Para avaliar capacidades multimodais abrangentes.
  - Métricas Específicas: mIoU (segmentação semântica), PSNR/SSIM (reconstrução de imagem/vídeo), AUC (classificação médica) e Success Rate (para tarefas criativas como receitas e memes).
  
- Métricas de Eficiência Computacional:
  - TTFT (Time To First Token): Mede a latência de início da geração.
  - Velocidade e Memória: Segundos por palavra (Sec./word) e ocupação de memória em inferência (Inf. Occ. em GB).
  - Contagem de Parâmetros e FLOPs: Para quantificar a redução de tamanho e operações.
  
## Resultados

Modelos compactos atingem resultados muito próximos a modelos de 7B/13B. Por exemplo, o LLaVA-Mini consegue manter desempenho comparável ao LLaVA-1.5 comprimindo os 576 tokens visuais para apenas 1 token visual.
As arquiteturas Mamba como o Cobra alcançam velocidades de inferência 3x a 4x mais rápidas que modelos como LLaVA-Phi, reduzindo parâmetros em 48% sem perder a precisão.
Os modelos como o FastVLM reduzem o tempo até o primeiro token (TTFT) para até 125ms-166ms, sendo até 7,9x mais rápidos que baselines como o Cambrian-1.

## Limitações

Apontadas pelo survey:
Modelos Mamba sem componentes específicos de normalização e mistura de canais apresentam instabilidade (o SiMBA propõe o EinFFT para estabilizar em escalas maiores).
Os benchmarks atuais podem não refletir o desempenho no mundo real, e o survey pede métricas de alinhamento multimodal e de robustez a dados ruidosos.
O survey lista como desafios abertos a eficiência de dados, a generalização entre domínios, vieses e falta de explicabilidade.

Não vêm do survey (busquei no texto e não aparecem; usar outra fonte antes de citar):
Perda de detalhes visuais finos por compressão extrema de tokens (o E3 apoia parcialmente: pixel shuffle de razão alta prejudica localização fina e OCR).
Taxa de alucinação alta, dificuldade com contagem exata e com relações espaciais em modelos menores (o E1 mostra falhas de relações semânticas no LLaVA; o E4 mostra queda de fidelidade com mais objetos na resposta).
Sensibilidade à qualidade dos dados de instrução (o E3 mostra efeito da composição de dados no SmolVLM).

## Ideias Reutilizaveis

### Taxonomia para Escolha dos Modelos (Configuração A e B)

- Taxonomia de Arquiteturas de sVLMs: O artigo categoriza os modelos compactos em três paradigmas principais: baseados em Transformer (ex.: Moondream2, TinyGPT-V, LLaVA-Mini), baseados em Mamba/SSM (ex.: VL-Mamba, Cobra) e Híbridos (ex.: FastVLM, SAMBA). O SmolVLM (E3) não aparece no survey.
- Como reutilizar, que permite justificar cientificamente a seleção do Small VLM. Pode comparar o comportamento da Configuração A e B em uma arquitetura baseada em atenção (Transformer) contra uma de complexidade linear (Mamba/SSM), analisando se o ganho de grounding varia entre paradigmas.

### Métricas Padronizadas de Eficiência Computacional

- O estudo compila métricas consolidadas na literatura para avaliar a eficiência de sVLMs em ambientes com restrição de recursos: TTFT ( Time To First Token ), segundos por palavra ( Sec./word ), ocupação de memória em inferência ( Inf. Occ. em GB) e FLOPs.
- Adotar exatamente essa bateria de métricas de eficiência no seu projeto para quantificar o overhead da Configuração B vs. Configuração A (medindo se a inclusão de texto estruturado altera a latência ou o consumo de memória do Small VLM de forma relevante).

### Direções de pesquisa listadas pelo survey

- O survey cita explicabilidade e interpretabilidade, avaliação além dos benchmarks atuais e geração de dados sintéticos como direções abertas. Isso apoia o enquadramento do projeto, mas não é evidência de que sVLMs alucinam mais.
- Para a justificativa da hipótese H1 (evidências estruturadas de CV como apoio factual), usar o E1 (falhas de relações semânticas), o E3 (limites de resolução e localização) e o E4 (queda de fidelidade com respostas longas e muitos objetos), e não este survey.

### Técnicas de Fine-Tuning Eficiente (LoRA e Adapters)

- O survey menciona LoRA e adapters apenas como direção de pesquisa para treino escalável (item 7 da seção 4), não como resultado.
- Se o Small VLM tiver dificuldade em usar as evidências em JSON na Configuração B, um ajuste leve com LoRA é uma opção a avaliar depois, com outra fonte de apoio.

## Relação com o projeto

O survey é uma referência muito boa para o mapeamento de Small VLMs, pois ele já fornece a taxonomia completa para selecionar e justificar qual Small VLM adotar como configuração A. Também fornece a lista estandardizada de métricas de eficiência (TTFT, latência, memória) e qualidade para usar na avaliação.
O survey não trata de alucinação nem de fidelidade; ele cobre arquitetura, eficiência e benchmarks de qualidade. A hipótese H1 (em vez de aumentar o modelo, injetar evidências estruturadas de modelos especialistas de Visão Computacional para elevar o FaithScore mantendo o baixo custo de inferência) precisa de apoio de outras referências, principalmente E1, E3 e E4. O papel do survey é situar o Small VLM escolhido na taxonomia e fornecer as métricas de eficiência.

