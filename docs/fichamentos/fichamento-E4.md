# Fichamento E4

Título: FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models (Jing, Li, Chen, Du, 2024, Findings of EMNLP 2024)

## Problema

LVLMs alucinam: geram conteúdo incompatível com a imagem, o que causa desinformação e problemas de segurança. As avaliações existentes têm duas fraquezas. Primeiro, focam em captioning ou em sondagem de objetos (CHAIR, POPE, NOPE) e não tratam respostas livres e longas a perguntas abertas. Segundo, ignoram alucinações finas de atributos, como contagem, cor e relações espaciais entre objetos. Além disso, uma resposta livre mistura conteúdo descritivo (verificável na imagem) e analítico (opinião, raciocínio, conhecimento externo), e misturar os dois distorce a medida de factualidade.

## Método

FaithScore (Faithfulness to Atomic Image Facts Score): métrica sem referência (não precisa de resposta de gabarito), com valor entre 0 e 1, calculada a partir da imagem I, do prompt Q e da resposta A. Três etapas, cada uma feita por um módulo:

1. Identificação de subsentenças descritivas (Recognizer): o ChatGPT divide a resposta em subsentenças por pontuação e rotula cada uma como descritiva ou analítica, com exemplos no prompt (in-context). Só as descritivas seguem.
2. Geração de fatos atômicos (Decomposer): o ChatGPT quebra cada subsentença descritiva em fatos atômicos mínimos, em 5 categorias: entity, relation, color, count e others.
3. Verificação (Verifier): um modelo de entailment visual (VEM) decide se a imagem sustenta cada fato (1 ou 0). Os autores escolheram o LLaVA-1.5 como verificador.

Fórmula: média dos escores dos fatos atômicos, com todos os pesos iguais a 1. Existe também uma versão em nível de sentença: 1 − Sh/S, em que S é o número de subsentenças descritivas e Sh o número das que têm alucinação.

## Dataset

- Meta-avaliação humana: LLaVA-Bench (baseado em COCO), 90 perguntas por tipo (descrição detalhada, conversa, pergunta complexa), com respostas do LLaVA e do InstructBLIP. Três anotadores seguem os mesmos passos da métrica (subsentença, fatos atômicos, verificação) e dão nota Likert de 1 a 5.
- LLaVA-1k: 1000 imagens da validação do MSCOCO, com 3 tipos de par prompt-resposta gerados por ChatGPT.
- MSCOCO-Cap: 1000 imagens da validação do MSCOCO, com o prompt "Generate a concise caption for the given image".

## Baseline

- Métricas comparadas na correlação com humanos: BLEU-4, ROUGE-L, METEOR, CHAIR, SPICE (com referência) e CLIPScore (sem referência).
- Modelos avaliados: MiniGPT-4, LLaVA, InstructBLIP, Multimodal-GPT, mPLUG-Owl e LLaVA-1.5. No apêndice, também GPT-4V e Gemini.

## Métricas

- Correlação com o julgamento humano: Pearson (r), Spearman (ρ) e Kendall (τ).
- Acurácia do Recognizer e do Verifier contra a anotação humana (voto majoritário).
- FaithScore e FaithScore em nível de sentença como medida de alucinação dos modelos.

## Resultados

- Correlação com humanos (Tabela 3): FaithScore com r 48,2%, ρ 38,4% e τ 47,6%. SPICE, a melhor das outras, teve 20,2%, 21,3% e 25,4%. BLEU-4, ROUGE-L e METEOR ficaram negativas.
- Recognizer: ChatGPT com 90,74% de acurácia contra 69,75% do LLaMA-7B.
- Verifier: LLaVA-1.5 teve a melhor acurácia (85,07%), seguido de mPLUG (84,35%).
- Alucinação nos modelos (Tabela 4, média em LLaVA-1k / MSCOCO-Cap): LLaVA-1.5 teve 0,8566 e 0,9425; InstructBLIP 0,8091 e 0,9392; LLaVA 0,8360 e 0,8729; mPLUG-Owl 0,7167 e 0,8546; MiniGPT-4 0,5713 e 0,6359; Multimodal-GPT 0,5335 e 0,5440. No apêndice: GPT-4V 0,8788 e Gemini 0,8865 no LLaVA-1k.
- Respostas mais longas e com mais objetos têm FaithScore menor. Exemplo: InstructBLIP vai de 0,895 com 1 objeto para 0,662 com 10 objetos. Captions curtas têm score melhor que as respostas longas do LLaVA-1k.
- Os tipos de fato atômico variam entre modelos; nenhum é consistentemente bom em todas as 5 categorias.
- Custo: cerca de 4 s por resposta longa, contra 212 s de anotação manual.

## Limitações

- Mede apenas precisão factual: não penaliza o modelo que diz pouco (sem recall). Os autores sugerem relatar o comprimento médio da resposta junto com o FaithScore.
- Depende da qualidade do verificador (85% de acurácia no melhor caso) e do ChatGPT para decompor; o Apêndice G traz erros na decomposição em fatos atômicos.
- Correlação moderada com humanos (r de 48,2%), melhor que as outras métricas, mas longe de perfeita.
- Distinguir descritivo de analítico depende de um julgamento, que o Recognizer acerta em cerca de 90% dos casos.
- A versão original usa o ChatGPT (API paga) nas etapas 1 e 2.

## Ideias reutilizáveis

1. Processo manual em três passos (descritivo ou analítico, fatos atômicos, verificação), que é a base do exp001. Posso reutilizá-lo como protocolo de anotação humana.
2. Separar conteúdo descritivo de analítico antes de medir factualidade.
3. As 5 categorias de fato (entity, relation, color, count, others) para classificar os erros do SmolVLM e ver onde as evidências do PatchCore ajudam.
4. Relatar FaithScore junto com o comprimento da resposta e o número de entidades.
5. Versão em nível de sentença como métrica complementar.
6. Testar verificadores alternativos (inclusive um Small VLM) e medir a concordância com minha anotação manual, como os autores fizeram com 7 VEMs.
7. Curva de FaithScore por número de objetos na resposta.

## Relação com o projeto

É a métrica principal para a hipótese de que evidências estruturadas reduzem alucinação. Ela vai comparar a Configuração A (imagem + pergunta) com a B (imagem + evidências + pergunta). Meu exp001 já seguiu a lógica fato a fato e serve para calibrar a confiança na versão automática.

Diferença metodológica a resolver: no exp001 marquei como "não verificável" afirmações especulativas ("conversando", "aproveitando a refeição"). O FaithScore as rotularia como analíticas e as tiraria do cálculo, em vez de contá-las como erro. Preciso decidir se o protocolo futuro separa descritivo de analítico, para que a avaliação manual e a automática sejam comparáveis. As categorias de fato (count, relation) também ajudam a medir se o erro cai em contagem e localização, que é o que as evidências do PatchCore deveriam corrigir.

Riscos práticos: o código original usa o ChatGPT na decomposição, o que gera custo e dependência de API; é preciso avaliar um decompositor aberto e o efeito disso na correlação.
