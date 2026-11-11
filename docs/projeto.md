# Projeto IC4: Pequenos Modelos Vision-Language para Explicação de Resultados de Visão Computacional

Estudante: Edryck. Duração: 12 meses (52 semanas), cerca de 10 h por semana. Artefato final: VisionExplain.

## Visão geral

Modelos de Visão Computacional (CV) produzem saídas estruturadas, como classes, caixas, máscaras, mapas de anomalia e escores de confiança, mas não as explicam em linguagem natural. Pequenos Vision-Language Models (Small VLMs) poderiam gerar essas explicações com baixo custo computacional, porém tendem a alucinar. Não está claro o quanto fornecer explicitamente as evidências estruturadas do modelo de CV melhora a fidelidade (groundedness) da explicação.

Pipeline investigado: imagem → modelo de CV → evidências estruturadas → Small VLM → explicação.

O projeto compara configurações controladas do mesmo Small VLM:

- Configuração A: imagem + pergunta (linha de base, sem evidências).
- Configuração B: imagem + relatório de evidências do modelo de CV + pergunta. O bloco da pergunta é idêntico ao da A.
- Configuração C (opcional): VLM maior, como referência de qualidade e custo. Até a semana 18 não foi executada por falta de GPU no ambiente local.

Domínio de aplicação atual: inspeção industrial, com detecção de anomalias no MVTec AD. O projeto é de CV + VLM em geral e não se restringe a esse domínio.

## Problema

Small VLMs são candidatos a explicar resultados de CV em ambientes com pouco recurso computacional, mas suas respostas podem conter afirmações não sustentadas pela imagem. Falta evidência experimental controlada sobre se, e quanto, as evidências estruturadas produzidas por um modelo de CV tornam essas explicações mais factuais, e a que custo.

## Pergunta de pesquisa

Pequenos Vision-Language Models conseguem produzir explicações fundamentadas nas evidências geradas por modelos de Visão Computacional, mantendo qualidade adequada e custo computacional menor que o de modelos multimodais maiores?

## Hipóteses

- H1: fornecer evidências estruturadas ao Small VLM (Configuração B) produz respostas mais factuais do que fornecer apenas a imagem (Configuração A), com o mesmo modelo e custo computacional semelhante.
- Hipótese do exp003: sem evidências, o SmolVLM responde com baixa factualidade sobre defeitos industriais.
- Resultado nulo admitido: "as evidências não ajudam", ou ajudam apenas por cópia do relatório, é um resultado possível e cientificamente válido. A crítica metodológica de E5 sustenta essa possibilidade: grounding nem sempre reduz alucinação de forma trivial.

## Objetivo geral

Investigar se Small Vision-Language Models produzem explicações mais fundamentadas e factuais quando recebem evidências estruturadas de um modelo de Visão Computacional, em comparação com um cenário sem essas evidências e, quando viável, com um modelo multimodal maior.

## Objetivos específicos

1. Montar o pipeline imagem → modelo de CV → evidências estruturadas → Small VLM → explicação.
2. Implementar e avaliar a Configuração A (Small VLM + imagem).
3. Implementar e avaliar a Configuração B (Small VLM + imagem + evidências).
4. Quando viável, implementar a Configuração C (VLM maior) como referência.
5. Medir factualidade, custo computacional e qualidade das respostas.
6. Desenvolver o artefato VisionExplain.

## Contribuição esperada

Evidência experimental controlada (A vs. B, com C opcional) sobre o efeito de evidências estruturadas na fidelidade de explicações geradas por Small VLMs. Inclui a análise de quando o ganho é compreensão visual e quando é apenas repasse do relatório.

## Artigos e artefatos

- Artigo 1: "Avaliação de Small Vision-Language Models para interpretação de resultados visuais". Escopo: mapeamento de Small VLMs, Configuração A como linha de base, comparação preliminar A vs. B, métricas de factualidade e custo computacional. Meta: mês 6. Antes da redação entra o experimento intermediário exp007 (A vs. B nas 15 categorias do MVTec AD), aprovado pelo orientador.
- Artigo 2: "Explicação fundamentada em evidências estruturadas produzidas por modelos de Visão Computacional". Perguntas: a Configuração B melhora factualidade, groundedness e consistência em relação à A? Em que tipo de evidência o ganho é maior? Meta: mês 11.
- Artefato VisionExplain: carregar a imagem, executar o modelo de CV, extrair as evidências, enviá-las ao Small VLM, gerar a explicação, apresentar as evidências usadas, comparar as explicações e registrar as métricas.
- Artefatos de pesquisa: código em `src/`, registros de experimento em `experiments/`, notebooks em `notebooks/`, relatórios das Fases 3 e 4 em `docs/`, e dashboard de acompanhamento (`index.html`).

## Portfólio bibliográfico atual

| ID | Autores (ano) | Título | Função no projeto |
|---|---|---|---|
| E1 | Liu, Li, Wu e Lee (2023) | Visual Instruction Tuning (LLaVA) | Fundamento de VLMs; candidato à Configuração C |
| E2 | Patnaik et al. (2025) | Small Vision-Language Models: A Survey on Compact Architectures and Techniques | Revisão de Small VLMs |
| E3 | Marafioti et al. (2025) | SmolVLM: Redefining small and efficient multimodal models | Modelo principal |
| E4 | Jing, Li, Chen e Du (2024) | FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models | Métrica de factualidade |
| E5 | autores a confirmar (2024) | Does Object Grounding Really Reduce Hallucination of Vision-Language Models? | Crítica metodológica ao grounding |
| E6 | Bergmann, Fauser, Sattlegger e Steger (2019) | MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection | Dataset principal |
| E7 | Bergmann, Batzner, Fauser, Sattlegger e Steger (2022) | Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization (MVTec LOCO AD) | Dataset alternativo (anomalias lógicas) |
| E8 | vikhyatk (2025) | Moondream2: model card | Small VLM opcional para comparação |

Existem fichamentos de E1, E2, E3, E4 e E6. E5 e E7 ainda não têm fichamento. O PatchCore, modelo de CV usado no projeto, ainda não está no portfólio como referência formal.

## Modelos

- Small VLM principal: `HuggingFaceTB/SmolVLM-Instruct` (E3), com decodificação gulosa, `max_new_tokens=128`, `longest_edge=1152`, cache do prefixo da imagem e batch 1.
- Modelo de CV produtor de evidências: PatchCore (biblioteca anomalib, backbone `wide_resnet50_2`), um modelo por categoria. Gera o veredito normal/anômalo, o escore da imagem e até 5 regiões, com posição na grade 3x3, área e escore de pico.
- Opcionais: Moondream2 (E8), e LLaVA ou outro VLM maior para a Configuração C (E1).

## Dados

- MVTec AD (E6): imagens industriais com 73 tipos de defeito e máscaras de referência em nível de pixel.
- Fase 4 (exp003 e exp005): categoria `bottle`, 40 imagens (10 por grupo: `broken_large`, `broken_small`, `contamination` e `good`), semente 42, arquivo `data/subset_avaliacao.json`.
- exp007 (intermediário): 15 categorias e 740 imagens. Cada categoria tem 50 imagens (10 normais e 40 anômalas, divididas entre os tipos de defeito), exceto `toothbrush`, que tem 40. Semente 42, arquivo `experiments/exp007_multicategoria/subset.json`.
- Perguntas fixas: q1 presença ("Is there any defect in this image?"), q2 localização ("Where is the defect located in the image?") e q3 tipo ("What type of defect is it?").
- Alternativos: MVTec LOCO AD (E7) e conjuntos gerais de VQA ou visual grounding, ainda a definir.

## Métricas

- Métrica principal: acerto por gabarito. Em q1, a resposta é comparada ao rótulo. Em q2, as células 3x3 citadas são comparadas com a máscara. Em q3, os termos citados são comparados com um dicionário por tipo de defeito.
- Métrica secundária: FaithScore simplificado (E4), com decomposição em afirmações por regras e o SmolVLM como verificador sim/não. Na validação (exp004), teve kappa de 0,602 e viés para "yes". No MVTec, 45 de 64 respostas erradas receberam FaithScore 1,0.
- Qualidade da evidência (exp007): AUROC de imagem e de pixel do PatchCore, acurácia de detecção, falsos positivos e falsos negativos, IoU das caixas com a máscara e acerto de célula 3x3.
- Fidelidade à evidência (exp007): concordância com o veredito do PatchCore, cópia literal da posição do relatório e respostas genéricas ("anomaly").
- Consistência entre perguntas (exp007): proporção de imagens em que q1, q2 e q3 se contradizem.
- Riqueza (exp007): tokens gerados, palavras, número de afirmações e taxa de respostas sem afirmação.
- Custo computacional: latência do VLM por pergunta, tempo do PatchCore por imagem e pico de memória (RSS/VRAM).
- Estatística: teste de McNemar exato pareado, por pergunta e por categoria, e bootstrap por imagem para o IC95% da diferença B − A.
- Avaliação complementar: análise qualitativa com galeria de pares A/B e categorias de alucinação.

## Riscos

1. A Configuração C pode exigir recursos indisponíveis. Ela é tratada como opcional; até agora não foi executada (o notebook local não tem GPU e o SmolVLM já usa cerca de 13,2 GB de RAM).
2. O FaithScore original depende de LLMs externos. A versão local é simplificada e se mostrou pouco confiável neste domínio, por isso passou a ser métrica secundária.
3. Resultado nulo: as evidências podem não ajudar, ou ajudar só por cópia do relatório, como observado em q2 na Fase 4 (29 de 30 respostas anômalas repetiram a posição do relatório).
4. A amostra é pequena e as respostas não são independentes: na Fase 4 foram 40 imagens com 3 perguntas cada. O exp007 amplia para 740 imagens e usa bootstrap por imagem.
5. Evidência sempre correta: no `bottle`, o PatchCore acertou as 40 imagens, então o efeito de evidência errada não foi medido. O exp007 busca categorias em que ele erra.
6. Respostas curtas: na Fase 4, a B gerou em média 3,9 tokens e deixou o FaithScore indefinido em 103 de 120 respostas. Isso conflita com a meta de gerar explicações.
7. Viés das perguntas: q2 pressupõe que há defeito, e o gabarito de q3 é heurístico e permissivo.
8. Tempo de execução: o exp007 roda em CPU no mesmo notebook da Fase 4 e leva cerca de 55 h de VLM. O Colab foi abandonado por problemas de ambiente.
