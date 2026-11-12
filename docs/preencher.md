# Preencher no dashboard: semanas 9 a 14

Textos prontos para colar, com números reais. Revise antes de salvar. Itens com **[VOCÊ]** são revisões que só você pode fazer. Datas de execução: 04/11/2026 e 05/11/2026; ajuste se registrar em outro dia.

## Conferências da Fase 3

Concluídas pelo Claude a seu pedido, com detalhes em `docs/auditoria_fase3.md`. Os números abaixo já estão corrigidos. **[VOCÊ]** Confirme o `exp001`: a revisão não é anotação humana.

## Experimentos

### exp003: Configuração A completa

- Hipótese: sem evidências, o SmolVLM responde com baixa factualidade sobre defeitos industriais.
- Pergunta: como o SmolVLM responde a presença, localização e tipo de defeito só com a imagem?
- Dataset: MVTec AD, `bottle`, 40 imagens (`data/subset_avaliacao.json`)
- Modelo: HuggingFaceTB/SmolVLM-Instruct
- Configuração: A (imagem + pergunta)
- Parâmetros: `do_sample=False`, `max_new_tokens=128`, float32, CPU, batch 1, `longest_edge=1152`, KV cache do prefixo
- Variáveis: tipo de defeito, pergunta
- Métricas: FaithScore simplificado, acerto por gabarito
- Resultado: 120 de 120 respostas. Acerto por gabarito 0,453 (n=117); q1 0,600, q2 0,405, q3 0,350. `contamination` 0,100 e `good` 0,767. FaithScore simplificado 0,797 (n=113).
- Análise: em q1, o modelo diz "Yes" em 18 de 30 anômalas e 5 de 10 normais, quase sem separar os grupos. Metade dos "Yes" em anômalas fala de blur ou reflexo. Em 14 de 40 imagens há contradição entre perguntas.
- Conclusão: a hipótese é sustentada neste subconjunto, com a ressalva de n pequeno.
- Próximo passo: rodar a Configuração B nas mesmas imagens e perguntas, com o mesmo `longest_edge=1152` e o mesmo cache.
- Status: Concluído

### exp004: validação da métrica

- Hipótese: um FaithScore simplificado com verificador local concorda com a avaliação manual.
- Pergunta: a métrica automática reproduz a avaliação do exp001?
- Dataset: 10 respostas e 49 afirmações do exp001 (fotos comuns)
- Modelo: SmolVLM-Instruct como verificador
- Configuração: decomposição por regras, verificador sim/não
- Parâmetros: sem image splitting, limiar 0,5
- Variáveis: afirmações manuais vs. automáticas
- Métricas: acurácia, kappa de Cohen, Pearson, Spearman
- Resultado: acurácia 0,878 (baseline 0,755), kappa 0,602. Aceitou 6 de 12 afirmações falsas e rejeitou 0 de 37 verdadeiras. Pearson 0,419 (manuais) e 0,957 (automáticas), n=10.
- Análise: concordância moderada, com viés para "yes". No MVTec, 45 de 64 respostas erradas receberam FaithScore 1,0.
- Conclusão: a métrica não é confiável como medida principal neste domínio.
- Próximo passo: usar o acerto por gabarito como principal e validar em imagens do MVTec.
- Status: Concluído

### Custo da Configuração A

- Hipótese: o custo de CPU é viável para o subconjunto.
- Pergunta: qual a latência e a memória da Configuração A?
- Dataset: 4 imagens (1 por grupo), pergunta q1
- Modelo: SmolVLM-Instruct
- Configuração: A, sem cache, 1 aquecimento, 3 repetições por imagem
- Parâmetros: float32, batch 1, `longest_edge=1152`, AMD Ryzen 5 3500U, 4 threads
- Variáveis: imagem
- Métricas: latência, pico de RSS
- Resultado: 109,1 s por pergunta (desvio 0,79); pico de 13.174,5 MB de RSS; carga de 6,56 s.
- Análise: não há GPU, então o valor é RSS e não VRAM.
- Conclusão: viável em lote, inviável para uso interativo nesta máquina.
- Próximo passo: repetir o mesmo protocolo na Configuração B.
- Status: Concluído

## Resultados

### Factualidade da Configuração A

- Título: Factualidade e custo da Configuração A
- Métricas: acerto por gabarito 0,453; FaithScore simplificado 0,797; latência 109,1 s; pico 13.174,5 MB de RSS
- Tabelas: `docs/relatorio_fase3.md`
- Gráficos: `experiments/figs/alucinacoes_configA/`
- Observações: 7 respostas sem FaithScore e 3 sem pontuação por gabarito.
- Arquivos: `experiments/results/exp003_factualidade_configA.json`, `experiments/results/custo_configA.json`
- Interpretação: o modelo não detecta defeitos de forma confiável só com a imagem, e a métrica automática esconde isso.
- Relação com a hipótese: define o baseline da comparação com a Configuração B.

## Diário

### 04/11/2026: Configuração A e métrica

- Atividade: rodei a Configuração A nas 120 perguntas, implementei e validei o FaithScore simplificado e medi o custo.
- Descoberta: a primeira execução levaria cerca de 6,8 h. Com `longest_edge=1152` e cache, levou cerca de 1h30. O cache deu respostas idênticas em 6 de 6 testes; a resolução mudou 1 de 6.
- Problema: FaithScore de 0,797 contra acerto de 0,453. O verificador aceita respostas erradas.
- Decisão: adotar `longest_edge=1152` e cache para A e B. Propor acerto por gabarito como métrica principal.
- Hipótese: as evidências do PatchCore devem ajudar mais em q2, em que o modelo é fraco.
- Resultado: `exp003`, `exp004`, `custo_configA.json`.
- Próximo passo: revisar a galeria e levar a pauta ao orientador.

### 04/11/2026: decisão sobre otimizações de inferência

- Atividade: interrompi a primeira execução da Configuração A, com 7 de 120 respostas, e a refiz do zero com duas otimizações. As 7 respostas foram descartadas.
- Descoberta: as imagens de 900x900 eram ampliadas para `longest_edge=1536`. Com 1152, o prompt caiu de cerca de 1566 para cerca de 928 tokens. O tempo por imagem, com 3 perguntas, caiu de cerca de 610 s para cerca de 136 s (4,5x).
- Problema: em CPU, a execução levaria cerca de 6,8 h, e a etapa completa cerca de 8 h, ocupando o note.
- Decisão: adotar `longest_edge=1152` (`LONGEST_EDGE` em `src/vlm.py`) e reaproveitar o KV cache do prefixo com a imagem entre as 3 perguntas (`generate_answers_shared_image`). Regras: A e B usam os mesmos valores e o mesmo caminho de geração; a medição de custo não usa cache; `exp001` e notebook 02 usaram 1536 e não são diretamente comparáveis ao `exp003`. Rejeitei `longest_edge=768` e desligar o splitting nas respostas, pelo risco de apagar defeitos pequenos, e GGUF/int8, porque troca a precisão e não achei benchmark do SmolVLM em CPU.
- Hipótese: 1152 preserva a informação útil, porque a imagem original tem 900 px.
- Resultado: teste em 2 imagens (6 perguntas): cache e sem cache deram respostas idênticas em 6 de 6. Entre 1152 e 1536, 5 de 6 ficaram iguais; uma mudou de "tear on the glass" para "hole in the middle". Portanto, a resolução é mudança de protocolo, e o efeito dela não foi medido além dessas 6 perguntas.
- Próximo passo: orientador aprovou por enquanto, desde que igual em A e B; rever na revisão do artigo e registrar no Artigo 1 (semana 21). Detalhes em `docs/protocolo_dataset.md`.

### 05/11/2026: análise qualitativa e relatório

- Atividade: categorizei as alucinações e consolidei a Fase 3.
- Descoberta: 5 categorias. As principais são defeito real negado (36) e referência a "facts" inexistentes (15).
- Problema: 14 de 40 imagens têm respostas contraditórias entre perguntas.
- Decisão: selecionar a galeria pelo gabarito, não pelo FaithScore.
- Hipótese: as evidências devem reduzir a negação de defeitos reais, a categoria mais frequente.
- Resultado: `experiments/figs/alucinacoes_configA/`, `docs/relatorio_fase3.md`.
- Próximo passo: Configuração B (semana 15).

## Problemas

### Inferência lenta em CPU

- Problema: cerca de 204 s por pergunta, ou 6,8 h para a Configuração A.
- Data: 04/11/2026
- Impacto: atrasa experimentos e ocupa o note
- Prioridade: Média
- Solução tentada: `longest_edge=1152` e KV cache do prefixo.
- Resultado: cerca de 136 s por imagem com 3 perguntas, 4,5x mais rápido.
- Status: Resolvido

### Métrica automática pouco confiável

- Problema: 45 de 64 respostas erradas receberam FaithScore 1,0. O verificador é o próprio SmolVLM.
- Data: 05/11/2026
- Impacto: pode esconder o efeito real da Configuração B
- Prioridade: Alta
- Solução tentada: criar o acerto por gabarito (`src/ground_truth_score.py`).
- Resultado: orientador aprovou como medida principal, a conferir na avaliação do artigo. Validação no MVTec e troca do verificador seguem a definir.
- Status: Investigando

### Pouca memória durante a execução

- Problema: pico de 13,2 GB em uma máquina de 17 GB, e o note suspendeu no meio da fila.
- Data: 04/11/2026
- Impacto: o note fica lento, e a suspensão parou a fila por horas, sem perda de dados
- Prioridade: Baixa
- Solução tentada: execução retomável, salvando por imagem.
- Resultado: execução concluída.
- Status: Resolvido

## Checkpoints mensais

Pelo cronograma: mês 1 = semanas 1 a 4 (Fase 1), mês 2 = semanas 5 a 8 (Fase 2), mês 3 = semanas 9 a 12 e mês 4 = semanas 13 e 14 (a Fase 3 termina no mês 4). Os textos dos meses 1 e 2 foram montados a partir do repositório e das suas respostas.

### Checkpoint do mês 1

- O que foi realizado? Criação do repositório e do ambiente. Leitura dos artigos para entender o que é um Small VLM, com fichamentos E1 e E2. Inferência com SmolVLM (`notebooks/01_smolvlm_inferencia.ipynb`). Avaliação manual da inferência em 10 respostas e 49 afirmações (`exp001`).
- O que não foi realizado? Tudo o que foi planejado foi feito. Fica pendente só a revisão final do `exp001`, que segue como rascunho.
- Por quê? Não se aplica.
- Qual foi o principal resultado? No `exp001`, 37 de 49 afirmações foram verificáveis (0,755), com média por resposta de 0,813.
- Qual hipótese foi confirmada/refutada? Nenhuma hipótese central foi testada. O `exp001` mostrou que o SmolVLM produz afirmações não verificáveis, como contagens e especulações, em fotos comuns.
- Qual problema surgiu? Nenhum.
- O que mudou no planejamento? Nada.
- Qual será o próximo passo? Escolher domínio e modelo de CV e construir a extração de evidências.
- O que precisa ser discutido com o orientador? Nada pendente.

### Checkpoint do mês 2

- O que foi realizado? Confirmação do MVTec AD com o orientador. Pipeline dataset → PatchCore (via `anomalib`) → evidências (`src/extract_evidence.py`, `exp002`). Templates de prompt A e B (`src/prompts.py`, notebook 02). Subconjunto de avaliação com 40 imagens e 3 perguntas (`data/subset_avaliacao.json`, `docs/protocolo_dataset.md`). Fichamentos E3, E4 e E6.
- O que não foi realizado? Fechar formalmente o modelo de CV, com justificativa. Na prática, o PatchCore via `anomalib` já é o modelo em uso.
- Por quê? O domínio foi confirmado primeiro, e o pipeline avançou com o PatchCore antes do registro da escolha.
- Qual foi o principal resultado? Pipeline de evidências funcionando: nas 5 imagens do `exp002`, o PatchCore classificou as 4 anômalas e a normal corretamente. O subconjunto ficou fechado com 120 pares imagem-pergunta por configuração.
- Qual hipótese foi confirmada/refutada? Nenhuma ainda. No teste do notebook 02, em 3 imagens, a Configuração B deu respostas mais curtas, e na imagem normal respondeu "No" enquanto a A inventou um reflexo como defeito. É só um indício.
- Qual problema surgiu? Nenhum.
- O que mudou no planejamento? Nada.
- Qual será o próximo passo? Rodar a Configuração A em todo o subconjunto.
- O que precisa ser discutido com o orientador? Confirmar o uso do PatchCore e do MVTec AD. O MVTec AD foi confirmado.

### Checkpoint do mês 3

- O que foi realizado? Configuração A completa (120 de 120 respostas), FaithScore simplificado, validação contra o `exp001`, factualidade e custo da Configuração A.
- O que não foi realizado? Validação da métrica em imagens do MVTec, que não estava no cronograma mas se mostrou necessária.
- Por quê? O cronograma previa validar com o `exp001`, e o problema da métrica só apareceu ao rodar no MVTec.
- Qual foi o principal resultado? Acerto por gabarito de 0,453 (n=117) e FaithScore simplificado de 0,797 (n=113). Custo de 109,1 s por pergunta e pico de 13.174,5 MB de RSS.
- Qual hipótese foi confirmada/refutada? Nenhuma ainda: a hipótese central depende da Configuração B. O baseline A mostra baixa factualidade neste subconjunto.
- Qual problema surgiu? Inferência lenta em CPU (cerca de 6,8 h previstas) e FaithScore não confiável: 45 de 64 respostas erradas receberam 1,0.
- O que mudou no planejamento? A Configuração A foi refeita com `longest_edge=1152` e KV cache do prefixo, 4,5x mais rápido; a Configuração B deve usar os mesmos ajustes. A resolução é mudança de protocolo: mudou 1 de 6 respostas no teste. Foi criado o acerto por gabarito, proposto como métrica principal.
- Qual será o próximo passo? Analisar as alucinações e consolidar a Fase 3.
- O que precisa ser discutido com o orientador? Métrica principal e aprovação das otimizações.

### Checkpoint do mês 4

- O que foi realizado? Semanas 13 e 14: categorização de alucinações, galeria de 10 figuras e relatório da Fase 3 (`docs/relatorio_fase3.md`).
- O que não foi realizado? Revisão humana das categorias e validação da métrica em imagens do MVTec.
- Por quê? As categorias vêm de regras e aguardam sua revisão. A validação depende da decisão do orientador.
- Qual foi o principal resultado? 5 categorias de alucinação, com defeito real negado (36 respostas) e referência a "facts" inexistentes (15) como principais. Em q1, o modelo diz "Yes" em 18 de 30 anômalas e 5 de 10 normais, quase sem separar os grupos.
- Qual hipótese foi confirmada/refutada? Nenhuma ainda, porque a hipótese central depende da Configuração B.
- Qual problema surgiu? 14 de 40 imagens têm respostas contraditórias entre perguntas. O FaithScore não serve para escolher os exemplos da galeria.
- O que mudou no planejamento? A galeria foi selecionada pelo gabarito, não pelo FaithScore como previa o cronograma.
- Qual será o próximo passo? Configuração B (semana 15).
- O que precisa ser discutido com o orientador? Pauta em `docs/relatorio_fase3.md`.

## Reunião com o orientador

Registre tudo em uma única reunião. Pauta e Dúvidas já podem ser salvas; Decisões e Próximos passos ficam para depois da conversa.

**Pauta:**

1. Métrica principal da comparação A vs. B
2. Validação da métrica em imagens do MVTec
3. Otimizações de inferência (`longest_edge=1152` e KV cache)
4. Máquina da Configuração B
5. Reformulação da pergunta q2
6. Verificador diferente do SmolVLM
7. Registro formal do PatchCore como modelo de CV

**Dúvidas:**

1. Posso usar o acerto por gabarito como métrica principal e o FaithScore simplificado como secundária, já que 45 de 64 respostas erradas tiveram FaithScore 1,0?
2. Devo anotar à mão cerca de 30 respostas do MVTec para validar a métrica, já que a validação atual usou fotos comuns?
3. As otimizações são aceitáveis no artigo? O cache manteve as respostas em 6 de 6 testes, mas a resolução mudou 1 de 6.
4. Rodo a Configuração B neste note, no Mac M4 ou no Colab? Trocar de máquina exige refazer a medição de custo da A.
5. A q2 pressupõe que há defeito. Devo reformular, mesmo que isso exija refazer a Configuração A, ou registrar como limitação?
6. Vale testar um verificador diferente do SmolVLM, ou isso foge do escopo do Artigo 1?
7. O PatchCore via `anomalib` fica como modelo de CV definitivo, ou devo avaliar uma evolução dele?

**Decisões:**

1. Métrica: acerto por gabarito como principal e FaithScore simplificado como secundária. A avaliação do artigo confere se faz sentido.
2. Validação com cerca de 30 respostas do MVTec: a definir.
3. Otimizações: aprovadas por enquanto, desde que sejam exatamente iguais nas duas configurações. A revisão do artigo decide se ficam.
4. Máquina: a Configuração B roda neste notebook. Outras máquinas podem ser avaliadas depois.
5. Reformulação da q2: a definir.
6. Verificador: a definir. Por enquanto, fica o SmolVLM.
7. Modelo de CV: PatchCore mantido por enquanto.

**Dúvidas:** seguem em aberto as questões 2, 5 e 6.

**Próximos passos:**

- Iniciar a Configuração B neste notebook, com `longest_edge=1152`, cache e a q2 atual.
- Levar a definição das questões 2, 5 e 6 para a próxima reunião.
- Rever métrica e otimizações na revisão do artigo.

## Observações das semanas 9 a 14

- Semana 9: Refeita com `longest_edge=1152` e KV cache do prefixo, após a primeira execução ser interrompida com 7 respostas. 120 de 120 respostas salvas. Respostas curtas (média de 11,4 tokens).
- Semana 10: Versão simplificada: decomposição por regras e SmolVLM como verificador. Kappa de 0,602, com viés para "yes" (aceitou 6 de 12 afirmações falsas). Validado em fotos comuns, não no MVTec.
- Semana 11: Acerto por gabarito de 0,453 e FaithScore de 0,797. A diferença levou a propor o gabarito como métrica principal. 45 de 64 respostas erradas tiveram FaithScore 1,0.
- Semana 12: 109,1 s por pergunta (desvio 0,79), em CPU e sem cache. Pico de 13.174,5 MB de RSS, não VRAM, porque não há GPU. Medido só com q1, em 4 imagens.
- Semana 13: 5 categorias, 10 figuras. Seleção pelo gabarito, não pelo FaithScore. Categorias por regras, pendentes de revisão humana.
- Semana 14: Relatório e pauta prontos. Pendentes: checkpoints no dashboard e reunião com o orientador.

## Como fazer as revisões

### Galeria

Abra as 10 figuras de `experiments/figs/alucinacoes_configA/` e confira em cada uma:

- [ ] A resposta está mesmo errada, olhando a imagem e o contorno vermelho?
- [ ] A categoria descreve bem o erro?
- [ ] Em "defeito inventado", o reflexo existe? Se existe, o erro é só chamá-lo de defeito.

Depois, confira se alguma categoria deveria ser unida a outra e se falta alguma. Anote as mudanças em `README.md` e troque o status para revisado.

### Acerto por gabarito

Abra `experiments/results/exp003_factualidade_configA.json`. Escolha 15 respostas: 5 de cada pergunta, misturando tipos de defeito e acertos e erros. Para cada uma, decida você se está certa, sem olhar a nota, e depois compare com `ground_truth_correct`.

Casos que valem entrar:

- q1 com "blur" ou "reflection" em imagem anômala: hoje contam como acerto, mesmo sem citar o defeito.
- q2 "The defect is on the glass.": hoje é indefinido.
- q3 "It is a hole.": hoje conta como erro.

Anote quantas das 15 discordam. Se forem muitas, ajuste as regras antes da Configuração B e recalcule.

### exp001

Abra `experiments/exp001_avaliacao_manual.json` junto com as imagens de `data/test_images/`. Para cada uma das 49 afirmações:

- [ ] Confirme se `verifiable` está certo, olhando a imagem.
- [ ] Corrija `note` quando mudar de ideia.
- [ ] Marque especulações, como "enjoying a meal", com uma nota. O FaithScore as trataria como analíticas, não como erro.

Ao terminar, recalcule `n_verifiable`, `verifiable_ratio` e `summary`, e troque `status` para `human_reviewed`. Se alguma marcação mudar, rode `python -m src.validate_metric` (cerca de 16 min) para atualizar o `exp004` e os números do relatório.

## Fichamentos

Nenhum fichamento novo é necessário para as semanas 9 a 14. Faltam E5 e E7, que devem ser feitos antes da semana 20. No fichamento E4, vale acrescentar que a versão implementada troca ChatGPT por regras e LLaVA-1.5 por SmolVLM.

---

# Fase 4: semanas 15 a 18

Os dados da Fase 3 acima foram mantidos. Relatório completo: `docs/relatorio_fase4.md`. Revise antes de colar.

## Experimentos

### exp005: Configuração B completa

- Hipótese: evidências estruturadas reduzem alucinações e aumentam o acerto em relação à Configuração A.
- Pergunta: as evidências do PatchCore melhoram as respostas do SmolVLM sobre presença, localização e tipo de defeito?
- Dataset: MVTec AD, `bottle`, 40 imagens (`data/subset_avaliacao.json`)
- Modelo: HuggingFaceTB/SmolVLM-Instruct + PatchCore
- Configuração: B (imagem + evidências + pergunta)
- Parâmetros: iguais aos da A: `do_sample=False`, `max_new_tokens=128`, float32, CPU, batch 1, `longest_edge=1152`, KV cache do prefixo
- Variáveis: presença das evidências
- Métricas: acerto por gabarito (principal), FaithScore simplificado (secundária), latência, memória
- Resultado: acerto de 0,733 contra 0,453 da A. q1 0,875 contra 0,600; q2 1,000 contra 0,405; q3 0,325 contra 0,350. Só A acertou em 6 pares e só B em 38.
- Análise: o ganho vem principalmente de copiar o relatório: 29 de 30 respostas de q2 em anômalas repetem a posição. q3 não melhora, porque o relatório não informa o tipo. Em 5 imagens anômalas, a B respondeu "No." apesar do relatório. As respostas têm média de 3,9 tokens.
- Conclusão: a hipótese é sustentada para presença e localização, não para tipo. O resultado mostra repasse da evidência, não melhor compreensão visual.
- Próximo passo: discutir com o orientador pergunta aberta de explicação e teste com evidência errada.
- Status: Concluído

### Custo da Configuração B

- Hipótese: as evidências acrescentam pouco custo.
- Pergunta: quanto a Configuração B custa a mais que a A?
- Dataset: 4 imagens (1 por grupo), pergunta q1
- Modelo: SmolVLM-Instruct + PatchCore
- Configuração: B, sem cache, 1 aquecimento, 3 repetições por imagem
- Parâmetros: mesmo protocolo e mesma máquina da A
- Variáveis: presença das evidências
- Métricas: latência, pico de RSS
- Resultado: 114,6 s por pergunta (desvio 2,07) contra 109,1 s; PatchCore 0,92 s por imagem; pico de 13.659,0 MB de RSS contra 13.174,5 MB.
- Análise: cerca de 5% a mais de latência e 3,7% a mais de memória. O prompt passa de 928 para 995 tokens.
- Conclusão: o custo extra é pequeno.
- Próximo passo: usar os valores na figura de custo vs. factualidade (semana 22).
- Status: Concluído

### exp006: Configuração C

- Hipótese: não se aplica.
- Pergunta: é viável rodar um VLM maior nesta máquina?
- Dataset: não executado
- Modelo: não executado
- Configuração: C
- Parâmetros: não se aplica
- Variáveis: não se aplica
- Métricas: não se aplica
- Resultado: não executada.
- Análise: sem GPU, com 17 GB de RAM, e o SmolVLM já usa cerca de 13,2 GB. Um modelo maior exigiria quantização.
- Conclusão: inviável nesta máquina.
- Próximo passo: decidir com o orientador se vale rodar 12 imagens em Colab ou API.
- Status: Concluído

## Resultados

### Comparação A vs. B

- Título: Factualidade e custo das Configurações A e B
- Métricas: acerto 0,453 (A) e 0,733 (B); latência 109,1 s e 114,6 s; pico 13.174,5 MB e 13.659,0 MB
- Tabelas: `experiments/results/tabela_A_vs_B.csv`
- Gráficos: `docs/artigo1/figs/comparacao_A_B/`
- Observações: FaithScore não comparável: a B tem só 17 respostas com afirmações. O PatchCore acertou as 40 imagens, então a B é um melhor caso.
- Arquivos: `experiments/results/exp005_factualidade_configB.json`, `experiments/results/custo_configB.json`, `experiments/results/comparacao_pareada_A_vs_B.json`
- Interpretação: as evidências aumentam o acerto com custo baixo, mas sobretudo porque o modelo repete o relatório.
- Relação com a hipótese: sustenta a hipótese para presença e localização. Não sustenta para tipo de defeito.

## Diário

### 05/11/2026: conferências e Configuração B

- Atividade: conferi `exp001`, gabarito e galeria. Gerei as evidências das 40 imagens e rodei a Configuração B.
- Descoberta: o PatchCore acertou as 40 imagens. A auditoria corrigiu 1 regra ("Answer: Yes.") e 1 marcação do `exp001`.
- Problema: o note suspendeu durante a execução, mesmo com bloqueio de suspensão. Nada se perdeu.
- Decisão: criar `src/configurations.py` e `--config A|B`, mantendo um pipeline único para A e B.
- Hipótese: evidências corretas aumentam o acerto em presença e localização.
- Resultado: `exp002b`, `exp005`, `docs/auditoria_fase3.md`.
- Próximo passo: comparar A e B.

### 06/11/2026: comparação e análise qualitativa

- Atividade: montei a tabela A vs. B, os testes pareados, a decisão da Configuração C e a galeria de pares.
- Descoberta: acerto de 0,733 contra 0,453, mas 29 de 30 respostas de q2 repetem o relatório. q3 não melhora, e as respostas da B têm 3,9 tokens em média.
- Problema: FaithScore indefinido em 103 de 120 respostas da B.
- Decisão: comparar A e B só pelo acerto por gabarito. Não rodar a Configuração C nesta máquina.
- Hipótese: as evidências ajudam no que elas informam, e não no que falta nelas, como o tipo.
- Resultado: `tabela_A_vs_B.csv`, `docs/relatorio_fase4.md`.
- Próximo passo: levar a pauta ao orientador e iniciar o Artigo 1 (semana 19).

## Problemas

### FaithScore não compara A e B

- Problema: 103 de 120 respostas da B são curtas demais para gerar afirmações.
- Data: 06/11/2026
- Impacto: a métrica secundária fica sem valor na comparação
- Prioridade: Média
- Solução tentada: usar só o acerto por gabarito na comparação.
- Resultado: comparação feita; a limitação será declarada no artigo.
- Status: Investigando

### Sem teste com evidência errada

- Problema: o PatchCore acertou as 40 imagens, então o efeito de evidência errada não foi medido.
- Data: 06/11/2026
- Impacto: o resultado da B é um melhor caso
- Prioridade: Alta
- Solução tentada: nenhuma; levar ao orientador.
- Resultado: pendente.
- Status: Aberto

### Note suspende durante execuções

- Problema: a suspensão parou a fila, mesmo com `systemd-inhibit`. Causa provável: tampa fechada, não confirmada.
- Data: 05/11/2026
- Impacto: atraso de horas, sem perda de dados
- Prioridade: Baixa
- Solução tentada: execução retomável e bloqueio de suspensão.
- Resultado: execução concluída.
- Status: Resolvido

## Observações das semanas 15 a 18

- Semana 15: 120 de 120 respostas, com os mesmos pares da A. PatchCore acertou as 40 imagens. Criados `src/build_evidences.py` e `src/configurations.py`.
- Semana 16: acerto de 0,733 contra 0,453. q2 sobe para 1,000 por cópia do relatório; q3 não melhora. Custo extra de cerca de 5%. FaithScore não comparável.
- Semana 17: Configuração C não executada: sem GPU e com pouca RAM. Decisão em `experiments/exp006_configC.json`.
- Semana 18: galeria de pares em `docs/artigo1/figs/comparacao_A_B/`, com casos em que B corrigiu, piorou e em que as duas erraram. Não há caso de evidência errada.

## Checkpoint das semanas 15 a 18

Pelo cronograma, as semanas 15 e 16 são do mês 4, e as 17 e 18 são do mês 5. O mês 5 só fecha na semana 20. Acrescente ao checkpoint do mês 4 o que for das semanas 15 e 16, e use o restante no mês 5 quando ele terminar.

- O que foi realizado? Configuração B completa, tabela A vs. B, testes pareados, decisão da Configuração C e galeria de pares.
- O que não foi realizado? Configuração C e teste com evidência errada.
- Por quê? A máquina não comporta um VLM maior, e o PatchCore não errou nenhuma imagem do subconjunto.
- Qual foi o principal resultado? Acerto de 0,733 na B contra 0,453 na A, com cerca de 5% a mais de latência.
- Qual hipótese foi confirmada/refutada? Sustentada para presença e localização, não para tipo de defeito.
- Qual problema surgiu? O ganho vem sobretudo de copiar o relatório; FaithScore indefinido em 103 de 120 respostas da B.
- O que mudou no planejamento? Comparação feita só pelo acerto por gabarito. Configuração C não executada.
- Qual será o próximo passo? Esqueleto do Artigo 1 (semana 19).
- O que precisa ser discutido com o orientador? Pauta em `docs/relatorio_fase4.md`.

## Reunião com o orientador

Crie um novo registro depois da próxima reunião. A pauta está em `docs/relatorio_fase4.md`.

## Fichamentos

Nenhum fichamento novo é necessário nas semanas 15 a 18. E5 e E7 seguem pendentes para a semana 20.

