# Fichamento E3

Título: SmolVLM: Redefining small and efficient multimodal models (Marafioti et al., 2025, arXiv:2504.05299)

## Problema

VLMs grandes entregam bom desempenho, mas exigem muito processamento e memória, o que limita o uso em celulares e dispositivos de borda. Os VLMs pequenos existentes normalmente copiam as escolhas de projeto dos modelos grandes, como tokenização extensa da imagem, e isso gasta memória de GPU sem necessidade. Exemplos citados: Qwen2-VL e InternVL 2.5 têm variantes de 1B a 2B, mas mantêm custo computacional alto. A pergunta central é quais escolhas de arquitetura, tokenização e dados funcionam de fato em escala pequena.

## Método

Família SmolVLM com três variantes e exploração sistemática de projeto.

- Arquitetura: encoder visual SigLIP + backbone de linguagem SmolLM2. A imagem é dividida em sub-imagens, codificada, reorganizada por pixel shuffle e projetada por um MLP para o espaço do LLM. Os tokens visuais são concatenados com os tokens de texto.
- Variantes:
  - SmolVLM-256M: SigLIP-B/16 (93M) + SmolLM2-135M.
  - SmolVLM-500M: SigLIP-B/16 (93M) + SmolLM2-360M.
  - SmolVLM-2.2B: SigLIP-SO400M (400M) + SmolLM2-1.7B.
- Achados de arquitetura (§2):
  - Encoder menor é melhor quando o LM é pequeno (com LM de 360M, o encoder grande deu +11,6% de desempenho com +66% de parâmetros).
  - Contexto longo ajuda: RoPE base de 10k para 273k; 16k tokens no 2.2B e 8k nos menores.
  - Pixel shuffle agressivo (razão 4) ajuda modelos pequenos, mas razões altas prejudicam tarefas de localização fina como OCR.
  - Divisão de imagem (image splitting) ajuda em imagens; média de frames prejudica em vídeo.
- Achados de treino por instrução (§3):
  - Tokens posicionais aprendidos superam strings de texto para a posição das sub-imagens (evitam o "OCR loss plague").
  - Prompt de sistema curto e marcadores de início e fim de mídia melhoram o desempenho, principalmente em vídeo.
  - Mascarar o prompt do usuário durante o SFT (treinar só nas respostas) melhora a generalização.
  - Reutilizar texto de SFT de LLMs (SmolTalk) piora o resultado (até -3,7% em vídeo e -6,5% em imagem).
  - Muito dado de Chain-of-Thought prejudica; só uma fração mínima (0,02% a 0,05%) ajudou.
  - Vídeos de duração média até cerca de 3,5 minutos no treino ajudam; acima disso o ganho diminui.
- Treino em dois estágios: visão e depois vídeo.

## Dataset

- Estágio de visão: nova mistura baseada nos dados de Laurençon et al. (2024), mais MathWriting. Cobre documentos, legendas, VQA (2% de raciocínio multi-imagem), gráficos, tabelas e raciocínio visual, com um pouco de texto puro para manter conhecimento geral, matemática e código.
- Estágio de vídeo: 14% de texto e 33% de vídeo. Fontes: LLaVA-video-178k, Video-STaR, Vript, ShareGPT4Video, Vista-400k, MovieChat, FineVideo; multi-imagem de M4-Instruct e Mammoth.
- Pesos, datasets e código foram liberados.

## Baseline

Comparação com modelos open source eficientes: Idefics-80B (de 18 meses antes), MolmoE-A1B-7B (1B de parâmetros ativos), InternVL2-2B, Qwen2VL-2B e 7B, MiniCPM-V2, Moondream2 e PaliGemma. Os números de leaderboard vêm do OpenCompass OpenVLM.

## Métricas

- Qualidade: acurácia em 9 benchmarks de imagem (OCRBench, AI2D, ChartQA, TextVQA, DocVQA, ScienceQA, MMMU, MathVista, MMStar) e 5 de vídeo (Video-MME, MLVU, MVBench, WorldSense, TempCompass), avaliados com VLMEvalKit.
- Eficiência: uso de RAM/VRAM em batch size 1 e 64 (os autores argumentam que RAM é melhor proxy de custo do que número de parâmetros), vazão em exemplos por segundo (A100 e L4) e tokens decodificados por segundo no navegador via WebGPU.

## Resultados

- Média nos benchmarks: 44,0% (256M), 51,0% (500M) e 59,8% (2.2B).
- RAM com batch 1: 0,8 GB, 1,2 GB e 4,9 GB, contra 27,7 GB do MolmoE-A1B-7B. Qwen2VL-2B usa 13,7 GB e InternVL2-2B usa 10,5 GB.
- O 256M supera o Idefics-80B, que é cerca de 300 vezes maior, na maioria dos benchmarks.
- 2.2B em destaque: OCRBench 72,9%, TextVQA 73,0%, DocVQA 80,0%, ScienceQA 89,6%, MathVista 51,5%, Video-MME 52,1%.
- Vazão na A100: 256M de 0,8 a 16,3 exemplos/s (batch 1 a 64); 2.2B de 0,6 a 1,7.
- 256M chega a 80 tokens/s no navegador, em um MacBook Pro M4 Max.
- Aplicações derivadas: ColSmolVLM, SmolDocling e BioVQA.

## Limitações

O artigo não tem seção de limitações. O que ele mesmo mostra:

- O 2.2B não vence em tudo: o InternVL2-2B tem ScienceQA de 94,1 contra 90,0 e MMStar de 49,8 contra 46,0; o Qwen2VL-2B tem AI2D de 74,7 contra 70,0 e ChartQA de 73,5 contra 68,8.
- Modelos pequenos perdem em benchmarks que exigem raciocínio linguístico forte (MMMU 29,0% no 256M contra 42,3% do Idefics-80B; AI2D 46,4% contra 56,3%).
- Pixel shuffle com razão alta degrada tarefas que precisam de localização precisa.
- Achados valem para a faixa de 135M a 2.2B e para a receita específica de dados; não há garantia de generalização.

Observações minhas, que o artigo não avalia:

- Não mede alucinação nem fidelidade da resposta à imagem; só acurácia em benchmarks.
- A eficiência reportada é em GPU. No meu notebook 01, em CPU, o SmolVLM-Instruct de 2,2B levou 58 min para 15 perguntas, então a eficiência de memória não implica rapidez em CPU.

## Ideias reutilizáveis

1. Estrutura de prompt: prompt de sistema curto ("visual agent... concise answers") e marcadores explícitos antes e depois da mídia. Serve de modelo para delimitar o bloco de evidências no prompt.
2. Usar RAM em vez de número de parâmetros como medida de custo na comparação entre Configuração A e B.
3. As três variantes (256M, 500M, 2.2B) permitem testar se o ganho das evidências muda com o tamanho do modelo.
4. Contexto de 16k tokens no 2.2B (8k nos menores) dá limite prático para o tamanho do JSON de evidências.
5. Mascarar o prompt do usuário no SFT, caso haja ajuste fino com evidências.
6. VLMEvalKit como ferramenta de avaliação reprodutível.
7. Pouco Chain-of-Thought em modelos pequenos: cuidado ao pedir raciocínio longo no prompt.

## Relação com o projeto

É o modelo base do projeto (Configuração A: imagem + pergunta). Define como o Small VLM recebe imagem e texto, o que orienta a injeção das evidências do PatchCore como texto estruturado na Configuração B: delimitadores claros, instruções curtas e JSON compacto para caber no contexto. O artigo não trata de alucinação, então a hipótese de que evidências estruturadas aumentam a fidelidade não é testada aqui. Ela precisa de uma métrica como o FaithScore (E4). Em MVTec AD as imagens são grandes, então vale checar quantas sub-imagens o processador gera e o custo disso em CPU.
