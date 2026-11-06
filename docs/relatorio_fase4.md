# Relatório da Fase 4: experimentação A vs. B (semanas 15 a 18)

Status: rascunho para revisão do aluno e do orientador. Todos os números vêm dos arquivos citados.

## Semana 15: Configuração B

- Evidências: `experiments/exp002b_evidencias_subconjunto.json`, geradas por `src/build_evidences.py`.
- Respostas: `experiments/exp005_configB_completo.json`, 120 de 120, com os mesmos pares de imagem e pergunta da A.
- PatchCore: acertou as 40 imagens, e todas as regiões das anômalas tocam a máscara. O limiar veio do ajuste no treino, não deste subconjunto.

## Semana 16: comparação (`experiments/results/tabela_A_vs_B.csv`)

| Medida | A | B |
|---|---|---|
| Acerto por gabarito | 0,453 (n=117) | 0,733 (n=120) |
| q1 presença | 0,600 | 0,875 |
| q2 localização | 0,405 | 1,000 |
| q3 tipo | 0,350 | 0,325 |
| `broken_large` | 0,517 | 0,667 |
| `broken_small` | 0,429 | 0,733 |
| `contamination` | 0,100 | 0,533 |
| `good` | 0,767 | 1,000 |
| FaithScore simplificado | 0,797 (n=113) | 0,647 (n=17) |
| Tokens gerados por resposta | 11,4 | 3,9 |
| Latência do VLM por pergunta | 109,1 s | 114,6 s |
| PatchCore por imagem | não usa | 0,92 s |
| Pico de RSS | 13.174,5 MB | 13.659,0 MB |

Teste de McNemar exato em 117 pares (`experiments/results/comparacao_pareada_A_vs_B.json`):

| Grupo | Só A acertou | Só B acertou | p |
|---|---|---|---|
| Geral | 6 | 38 | 0,000001 |
| q1 | 3 | 14 | 0,013 |
| q2 | 0 | 22 | menor que 0,000001 |
| q3 | 3 | 2 | 1,000 |

Os pares vêm de 40 imagens, com 3 perguntas cada, e não são independentes. O p geral é só indicativo; os valores por pergunta são mais adequados.

## Semana 17: Configuração C

Não executada (`experiments/exp006_configC.json`). A máquina não tem GPU, o SmolVLM já usa cerca de 13,2 GB de RAM, e um modelo maior exigiria quantização. O orientador manteve este notebook por enquanto. Falta decidir se vale rodar 12 imagens em Colab ou API.

## Semana 18: análise qualitativa (`docs/artigo1/figs/comparacao_A_B/`)

Galeria de pares, com um exemplo por caso e tipo de defeito. Contorno vermelho: máscara. Caixa ciano: região do PatchCore. Pares no total: 38 em que B corrigiu A, 6 em que B piorou A e 26 em que as duas erraram.

## Achados

1. As evidências aumentam o acerto, mas o ganho vem principalmente de copiar o relatório. Em q2, 29 de 30 respostas em anômalas são a posição do relatório, palavra por palavra. Isso mostra repasse de informação, não melhor compreensão visual.
2. q3 não melhora. O relatório não informa o tipo, e o modelo responde "anomaly" ou "anomaly detection" em vez de um tipo. Das 6 vezes em que A disse "Crack.", 3 viraram resposta genérica.
3. A B ignora o relatório às vezes: em 5 imagens anômalas, respondeu "No." em q1 mesmo com o relatório apontando anomalia. Foram 4 de `contamination` e 1 de `broken_large`.
4. A B deixa as respostas pobres: média de 3,9 tokens. Ela elimina as alucinações de A, como as citações de "facts", mas quase não explica nada. Isso conflita com a meta de gerar explicações.
5. O custo extra é pequeno: cerca de 5% de latência e 3,7% de memória.

## Limitações

- O PatchCore acertou tudo, então não foi possível testar o que ocorre quando a evidência está errada. O resultado da B é um melhor caso.
- O FaithScore não serve para comparar A e B: 103 das 120 respostas da B são curtas demais para gerar afirmações.
- As perguntas pedem respostas curtas. O efeito das evidências sobre explicações mais longas não foi medido.
- Uma categoria, 10 imagens por grupo e regras de palavras-chave no gabarito.
- As auditorias foram feitas pelo Claude e ainda precisam da confirmação do aluno.
- As regras de gabarito não foram auditadas à mão na B, mas todos os padrões de resposta foram listados e conferidos.

## Pauta para a próxima reunião

1. O ganho da B é principalmente cópia do relatório. Como isso deve aparecer no Artigo 1?
2. Vale acrescentar uma pergunta aberta de explicação, já que a B responde com poucas palavras?
3. Como testar evidência errada: injetar relatórios incorretos em algumas imagens?
4. Configuração C: rodar 12 imagens em Colab ou API, ou encerrar?
5. Pendências: validação manual no MVTec, reformulação da q2 e verificador.
