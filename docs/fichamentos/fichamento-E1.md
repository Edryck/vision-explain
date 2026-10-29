# Fichamento E1

Título: Visual Instruction Tuning (LLaVA) (Liu et al., 2023, NeurIPS 2023, arXiv:2304.08485)

## Problema

Enquanto LLMs evoluíram rápido no alinhamento por instruções em texto, faltavam conjuntos de dados instrucionais multimodais (imagem-texto) para alinhar a visão ao comportamento de seguir instruções humanas. Os modelos de visão anteriores eram especialistas e resolviam tarefas isoladas, como classificação ou detecção, com interfaces rígidas e pouca adaptação às preferências dos usuários. Houve uma falta de alinhamento conversacional, os modelos multimodais treinados apenas em pares simples de imagem e legenda (como BLIP-2 e OpenFlamingo) focavam em só descrever cenas passivamente, sem conseguir interpretar nem responder a instruções complexas do usuário.

## Método

O artigo propõe estender o ajuste por instruções para o espaço multimodal de imagem e linguagem, utilizando uma arquitetura unificada e leve, composta por um encoder visual, decoder de linguagem e um conector de projeção.
- Encoder visual: utiliza o CLIP ViT-L/14 pré-treinado para extrair atributos da imagem.
- Decoder de linguagem: adota o Vicuna como LLM base, devido às suas capacidades conversacionais.
- Conector de projeção: utiliza uma simples matriz de projeção linear W para mapear os tokens visuais diretamente no mesmo espaço de representação de texto do LLM.
O treinamento foi dividido em duas etapas:
- Etapa 1:
  - Pré-treinamento para alinhamento de atributos: usa um subconjunto filtrado de 595 mil pares imagem-texto do CC3M (CC-595K). O encoder visual e o LLM permanecem congelados, treinando apenas a matriz de projeção.
- Etapa 2:
  - Ajuste fino ponta a ponta: mantém apenas o encoder visual congelado e atualiza a camada de projeção e o LLM usando dados instrucionais.

## Dataset

- LLaVA-Instruct-158K: Como o GPT-4 da época aceitava apenas entradas de texto, os autores converteram imagens do dataset COCO em representações simbólicas (legendas e caixas delimitadoras com coordenadas espaciais) e enviaram ao GPT-4.
- 158.000 amostras sintéticas, divididas em 3 modalidades:
- Conversação: 58.000 diálogos multirrodada cobrindo identificação de objetos, contagem, ações e posições relativas.
- Descrição Detalhada: 23.000 amostras com descrições exaustivas e abrangentes das imagens.
- Raciocínio Complexo: 77.000 perguntas em profundidade exigindo raciocínio lógico passo a passo.

## Baseline

OpenFlamingo e BLIP-2 (Modelos Multimodais Existentes): ao avaliar a capacidade de conversação sobre imagens do mundo real (como o exemplo clássico de um homem passando roupa na traseira de um táxi), modelos como BLIP-2 e OpenFlamingo focavam predominantemente em apenas descrever passivamente a cena, falhando em seguir as instruções diretas do usuário.
LLaVA-Bench (In-the-Wild): O LLaVA demonstrou uma capacidade superior de seguir instruções e raciocinar sobre o contexto visual, superando o BLIP-2 em +29% e o OpenFlamingo em +48% na pontuação relativa do benchmark.
GPT-4 de Texto Puro (Como Limite Superior Teórico): para criar uma métrica quantitativa no LLaVA-Bench, utilizou-se o GPT-4 de texto alimentado com descrições perfeitas da imagem (legendas e caixas delimitadoras) como uma referência de limite superior teórico. O LLaVA alcançou 85,1% de desempenho relativo em relação a esse limite no conjunto COCO.
Baselines no ScienceQA:
- O LLaVA foi comparado com métodos como GPT-3.5 (com e sem Chain-of-Thought — 75,17%), LLaMA-Adapter (85,19%) e o estado da arte até então, MM-CoT (91,68%).
- Sozinho, o LLaVA atingiu 90,92%, e quando combinado em ensemble sob a arbitragem do GPT-4, estabeleceu um novo recorde de 92,53% de precisão.

## Métricas utilizadas no LLaVA

Para medir a capacidade do assistente em seguir instruções humanas multimodais, os autores definiram duas abordagens principais:
- Pontuação Relativa via GPT-4 (LLaVA-Bench):
  * Como não existia uma métrica automática tradicional capaz de avaliar respostas abertas em conversas multimodais, os autores utilizaram o GPT-4 de texto puro como juiz.
  * O GPT-4 recebeu a pergunta, as informações visuais (representadas por descrições e coordenadas perfeitas) e as respostas geradas pelos modelos.
  * O juiz atribuiu uma nota de 1 a 10 avaliando quatro critérios centrais: utilidade (helpfulness), relevância (relevance), acurácia (accuracy) e nível de detalhamento (level of detail).
  * A Pontuação Relativa (%) foi calculada comparando a nota do LLaVA com o desempenho do próprio GPT-4 de texto (que atuou como limite superior teórico) nos conjuntos LLaVA-Bench (COCO) e LLaVA-Bench (In-the-Wild).
- Acurácia Percentual Direta (%) (ScienceQA):
  * No benchmark de múltipla escolha ScienceQA, a métrica principal foi a Acurácia (%).
  * A acurácia foi detalhada por assuntos (Ciências Naturais, Ciências Sociais, Linguagem), por modalidade de contexto (Texto, Imagem, Sem contexto) e por nível escolar (1º ao 6º ano, 7º ao 12º ano).

## Resultados

- LLaVA-Bench (COCO): O modelo alcançou uma pontuação relativa de 85,1% em comparação com o GPT-4 de texto (que recebia descrições textuais perfeitas das imagens).
- LLaVA-Bench (In-the-Wild): Superou significativamente modelos como BLIP-2 (+29%) e OpenFlamingo (+48%), atingindo 81,7% em raciocínio complexo.
- ScienceQA:
  - Sozinho, o LLaVA alcançou 90,92% de acurácia no benchmark multimodal ScienceQA.
  - Quando combinado em ensemble com o GPT-4 de texto (usando o GPT-4 como juiz em caso de divergência), estabeleceu um novo estado da arte (SoTA) de 92,53% de acurácia.
- Capacidades Emergentes: O assistente demonstrou habilidades de raciocínio não vistas no treino, como explicar o humor de memes, realizar OCR básico, gerar código HTML/JS a partir de rascunhos desenhados à mão e reconhecer personalidades/obras de arte (como Elon Musk e a Mona Lisa) através da generalização do LLM.

## Limitações

O LLaVA pode responder com afirmações incorretas sobre elementos que não existem na imagem. Os autores ilustram uma falha marcante em que o modelo respondeu "sim" quando perguntado se havia iogurte de morango dentro de uma geladeira, quando na verdade havia apenas iogurte natural e morangos frescos separados. Isso mostrou que o modelo às vezes percebe a imagem como um "saco de retalhos" de conceitos soltos, falhando em capturar as relações semânticas complexas entre os objetos.
Como o encoder visual CLIP trabalha com resolução fixa e limitada, o LLaVA tem dificuldades para ler rótulos pequenos, identificar marcas específicas (como a marca de um produto ao fundo) ou processar detalhes visuais de alta resolução.
Para responder a certas perguntas de domínio aberto no mundo real, o assistente ainda necessita de uma cobertura de conhecimento mais abrangente e maior capacidade de raciocínio multilíngue (por exemplo, identificar o nome de um restaurante específico de ramen a partir do formato dos pratos e sinalizações).
O LLaVA herda diretamente os vieses conceituais, sociais e limitações presentes nas suas redes base: o encoder de visão (CLIP) e o modelo de linguagem (LLaMA/Vicuna).
Medir com precisão o nível de alucinação e a fidelidade visual é uma tarefa complexa que exige métodos de avaliação mais robustos. Além disso, para evitar respostas nocivas ou o processamento de imagens inadequadas, o modelo precisa contar com filtros externos de segurança (como o OpenAI Filter API para texto e filtros NSFW para imagens).

## Ideias reutilizaveis

1. Pipeline de Geração Sintética de Dados: Usar conversão simbólica (texto + coordenadas) e LLMs fortes para sintetizar dados de instrução em domínios especializados (ex.: medicina, documentos) ou novas modalidades (ex.: áudio, sensores).
2. Treinamento em Duas Etapas: Congelar o encoder visual e o LLM na 1ª etapa (alinhamento) e ajustar na 2ª etapa para economizar GPU, acelerar o treino e evitar o esquecimento catastrófico.
3. Conectores Modulares e Leves: Adotar projeções lineares simples para conectar a visão ao LLM, permitindo a troca ágil de backbones (visuais ou de linguagem) e a prototipagem rápida.
4. Avaliação com LLM como Juiz (LLM-as-a-Judge): Empregar LLMs avançados para automatizar a medição de qualidade, relevância e acurácia de respostas abertas multimodais em benchmarks internos.
5. Arbitragem e Ensemble Inteligente: Combinar predições de modelos especialistas e LLMs com arbitragem para elevar a precisão e validar respostas críticas antes da entrega ao usuário.
6. Adaptação para Dispositivos de Borda (sVLMs): Aplicar a sintonização por instruções em modelos ultraleves, utilizando compressão de tokens visuais ou arquiteturas alternativas (como Mamba/SSM) para execução em hardware limitado.

## Relação com o projeto

Serve como fundamentação teórica, metodológica e justificativa de problema: ele fornece a prova de conceito de que modelos de linguagem conseguem processar representações simbólicas em texto (como caixas delimitadoras e classes de CV) para fundamentar o raciocínio espacial, documenta formalmente a limitação do "bag of patches" e as alucinações visuais que justificam a necessidade de ancorar Small VLMs em evidências estruturadas, e oferece a estrutura de prompts, o pipeline de integração entre visão especialista e linguagem e as baselines de referência para a escrita dos artigos do projeto.
