# Plano manual: semanas 15 a 22

Ordem de execução: 15, 16, 17, 18, 19, 20, 21, 22. Rode as semanas 15 e 16 na mesma máquina, com a mesma precisão e o mesmo `LONGEST_EDGE` da Configuração A. Antes de começar, leia os resultados em `docs/relatorio_fase3.md` e as decisões da reunião da semana 14.

Decisões que afetam este plano: a B roda neste notebook; acerto por gabarito é a métrica principal, então use-o primeiro na tabela da semana 16; a q2 segue igual por enquanto. Se ela mudar depois, A e B precisam ser refeitas.

## Semana 15: Configuração B em todo o subconjunto

Entrega: `experiments/exp002b_evidencias_subconjunto.json` e `experiments/exp005_configB_completo.json`.

- [ ] Criar `src/build_evidences.py`: carregar o PatchCore uma vez com `load_patchcore()` e chamar `extract_evidence` nas 40 imagens de `data/subset_avaliacao.json`.
- [ ] Salvar a evidência de cada imagem com o `id` do subconjunto como chave.
- [ ] Conferir se o PatchCore marcou as 10 imagens `good` como normais e as 30 anômalas como anômalas. Anotar falsos positivos e falsos negativos.
- [ ] Calcular a concordância da região com a máscara (IoU ou célula 3x3). Sem isso, não dá para separar evidência certa de evidência errada.
- [ ] Em `src/run_configuration.py`, criar `prompt_builder_b(question, image)` usando `build_prompt_b` e a evidência da imagem.
- [ ] Adicionar um argumento `--config A|B` e rodar `run_configuration("exp005", "B", prompt_builder_b, "experiments/exp005_configB_completo.json")`.
- [ ] Testar antes com 1 imagem: o bloco `Question:` tem de ser igual ao da A e o restante do prompt tem de ser só a evidência.
- [ ] Conferir: 120 respostas e os mesmos pares `(image_id, question_id)` do `exp003`.

Critério de conclusão: 100% das respostas salvas, com o mesmo conjunto de pares da Configuração A.

Armadilha: o prompt da B é mais longo, então `prompt_tokens` cresce. Isso faz parte do custo real da B.

## Semana 16: factualidade, custo e tabela A vs. B

Entrega: `experiments/results/exp005_factualidade_configB.json`, `experiments/results/custo_configB.json` e `experiments/results/tabela_A_vs_B.csv`.

- [ ] Rodar `python -m src.evaluate_factuality --input experiments/exp005_configB_completo.json --output experiments/results/exp005_factualidade_configB.json`.
- [ ] Em `src/measure_cost.py`, adicionar `prompt_builder_b`, mantendo imagens, repetições e aquecimento.
- [ ] Rodar o custo da B com o computador ocioso, sem outros programas abertos.
- [ ] Somar ao custo da B o tempo do PatchCore por imagem. Ele faz parte do pipeline.
- [ ] Criar `src/compare_configurations.py` para gerar o CSV com: FaithScore simplificado, acerto por gabarito (q1, q2, q3), tamanho médio da resposta, latência e memória.
- [ ] Separar os resultados por tipo de defeito e por evidência certa ou errada.
- [ ] Como os pares são os mesmos, usar teste pareado (McNemar no acerto por gabarito, Wilcoxon no FaithScore) e relatar o tamanho do efeito.
- [ ] Revisar à mão 10 linhas da tabela contra os JSONs de origem.

Critério de conclusão: tabela com valores reais de factualidade e custo das duas configurações.

Armadilha: respostas curtas como "Yes." geram 0 fatos e FaithScore indefinido. Relate quantas são em cada configuração; a B tende a respostas mais curtas (notebook 02), o que pode distorcer a média.

## Semana 17: Configuração C (se viável)

Entrega: `experiments/exp006_configC.json` ou nota de decisão no Diário.

- [ ] Limitar a decisão a 1 dia de trabalho.
- [ ] Listar opções: VLM maior local, Colab com GPU ou API. Anotar memória, custo e licença.
- [ ] Decidir com o orientador se uma API paga é aceitável.
- [ ] Se viável, rodar em 12 imagens (3 por grupo), com as mesmas 3 perguntas e sem evidências.
- [ ] Avaliar com o mesmo `src/ground_truth_score.py`.
- [ ] Se inviável, registrar o motivo no Diário e seguir.

Critério de conclusão: decisão registrada e, se aplicável, resultado parcial.

## Semana 18: análise qualitativa A vs. B

Entrega: `docs/artigo1/figs/comparacao_A_B/` com a galeria comentada.

- [ ] Selecionar pelo menos 8 pares da mesma imagem e pergunta, com critério escrito antes de olhar os resultados.
- [ ] Cobrir 4 casos: B corrigiu A, B piorou A, os dois erraram, evidência errada do PatchCore.
- [ ] Reaproveitar o formato da galeria da semana 13: imagem, máscara, resposta A, resposta B, evidência, comentário.
- [ ] Usar as categorias de alucinação da semana 13 para classificar os dois lados.
- [ ] Anotar se a B só repete a evidência ou se acrescenta algo visual.

Critério de conclusão: 8 pares ou mais, incluindo casos em que a B não melhorou.

## Semana 19: consolidar e criar o esqueleto do Artigo 1

Entrega: `docs/artigo1/artigo1.md`.

- [ ] Criar `docs/artigo1/` e copiar as tabelas e figuras finais para um único lugar.
- [ ] Criar as seções: Introdução, Trabalhos relacionados, Metodologia, Experimentos, Resultados, Discussão, Conclusão, Referências.
- [ ] Escrever em cada seção 3 a 5 tópicos do que ela vai afirmar.
- [ ] Verificar o formato exigido pelo evento ou revista alvo.
- [ ] Preencher o checklist inicial do Artigo 1 no dashboard.
- [ ] Deixar claro no texto que A vs. B é uma avaliação preliminar.

Critério de conclusão: documento com todas as seções nomeadas.

## Semana 20: Introdução e Trabalhos relacionados

Entrega: seções em `docs/artigo1/artigo1.md`.

- [ ] Introdução: problema, por que usar VLMs pequenos, pipeline CV → evidências → VLM, pergunta de pesquisa e contribuições.
- [ ] Citar E1, E2 e E3 como motivação e E5 já na introdução.
- [ ] Trabalhos relacionados por tema: Small VLMs, alucinação e factualidade, detecção de anomalias.
- [ ] Conferir se E5 e E7 têm fichamento. Hoje existem E1, E2, E3, E4 e E6. Fazer os que faltarem antes de citar.
- [ ] Marcar as referências usadas no dashboard.

Critério de conclusão: pelo menos 5 referências verificadas citadas.

## Semana 21: Metodologia

Entrega: seção em `docs/artigo1/artigo1.md`.

- [ ] Descrever o subconjunto: 40 imagens, 4 grupos, 3 perguntas, semente 42.
- [ ] Descrever o PatchCore, o limiar e a conversão do mapa em regiões.
- [ ] Incluir o texto exato dos prompts A e B.
- [ ] Descrever geração: decodificação gulosa, `max_new_tokens`, `LONGEST_EDGE`, cache do prefixo, precisão e hardware.
- [ ] Descrever as duas métricas: FaithScore simplificado (e o que difere do original) e acerto por gabarito.
- [ ] Relatar a validação da semana 10 com os números reais, mesmo que sejam ruins.
- [ ] Pedir a alguém que leia e diga se conseguiria reproduzir.

Critério de conclusão: dataset, pipeline e métrica descritos com detalhe para reprodução.

## Semana 22: Resultados

Entrega: `docs/artigo1/figs/` e seção em `docs/artigo1/artigo1.md`.

- [ ] Tabela final A vs. B (e C, se houver), com n e medida de dispersão.
- [ ] Gráfico de dispersão de custo vs. factualidade.
- [ ] Acerto por pergunta e por tipo de defeito.
- [ ] Galeria qualitativa com legendas.
- [ ] Texto descritivo de cada tabela e figura, sem interpretar: a interpretação vai na discussão (semana 23).
- [ ] Conferir se cada figura citada existe como arquivo.

Critério de conclusão: todas as figuras citadas existem como arquivo com legenda.


1. Métrica principal da comparação A vs. B
2. Validação da métrica em imagens do MVTec
3. Otimizações de inferência (longest_edge=1152 e KV cache)
4. Máquina da Configuração B
5. Reformulação da pergunta q2
6. Verificador diferente do SmolVLM
7. Registro formal do PatchCore como modelo de CV

1. Posso usar o acerto por gabarito como métrica principal e o FaithScore simplificado como secundária, já que 45 de 64 respostas erradas tiveram FaithScore 1,0?
2. Devo anotar à mão cerca de 30 respostas do MVTec para validar a métrica, já que a validação atual usou fotos comuns?
3. As otimizações são aceitáveis no artigo? O cache manteve as respostas em 6 de 6 testes, mas a resolução mudou 1 de 6.
4. Rodo a Configuração B neste note, no Mac M4 ou no Colab? Trocar de máquina exige refazer a medição de custo da A.
5. A q2 pressupõe que há defeito. Devo reformular, mesmo que isso exija refazer a Configuração A, ou registrar como limitação?
6. Vale testar um verificador diferente do SmolVLM, ou isso foge do escopo do Artigo 1?
7. O PatchCore via anomalib fica como modelo de CV definitivo, ou devo avaliar uma evolução dele?