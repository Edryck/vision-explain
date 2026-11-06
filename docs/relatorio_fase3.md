# Relatório da Fase 3: reprodução da Configuração A (semanas 9 a 14)

Status: rascunho para revisão do aluno e reunião com o orientador. Todos os números vêm dos arquivos citados.

## Configuração

SmolVLM-Instruct, float32, CPU AMD Ryzen 5 3500U (4 threads), `batch_size` 1, decodificação gulosa, `max_new_tokens` 128, `longest_edge` 1152, KV cache do prefixo da imagem. Subconjunto: 40 imagens de `bottle` e 3 perguntas, ou 120 respostas. Detalhes em `docs/protocolo_dataset.md`.

## Resultados

### Semana 9: respostas (`experiments/exp003_configA_completo.json`)

120 de 120 respostas salvas, com média de 11,4 tokens gerados e máximo de 23.

### Semana 10: validação da métrica (`experiments/exp004_validacao_metrica.json`)

Métrica: FaithScore simplificado, com decomposição por regras e o próprio SmolVLM como verificador (sim ou não, sem splitting, limiar 0,5). Referência: 49 afirmações do `exp001`, revisadas em `docs/auditoria_fase3.md`.

| Medida | Valor |
|---|---|
| Acurácia por afirmação | 0,878 (baseline majoritário 0,755) |
| Kappa de Cohen | 0,602 |
| Afirmações falsas aceitas | 6 de 12 |
| Afirmações verdadeiras rejeitadas | 0 de 37 |
| Pearson por resposta, afirmações manuais | 0,419 (n=10) |
| Pearson por resposta, afirmações automáticas | 0,957 (n=10) |

### Semana 11: factualidade (`experiments/results/exp003_factualidade_configA.json`)

| Grupo | FaithScore simplificado | Acerto por gabarito |
|---|---|---|
| Geral | 0,797 (n=113) | 0,453 (n=117) |
| q1 presença | 0,872 | 0,600 |
| q2 localização | 0,625 | 0,405 |
| q3 tipo | 0,912 | 0,350 |
| `broken_large` | 0,963 | 0,517 |
| `broken_small` | 0,808 | 0,429 |
| `contamination` | 0,600 | 0,100 |
| `good` | 0,833 | 0,767 |

Indefinidos: 7 no FaithScore (respostas curtas, como "Crack.", sem afirmação) e 3 no gabarito.

### Semana 12: custo (`experiments/results/custo_configA.json`)

| Medida | Valor |
|---|---|
| Latência por pergunta isolada, sem cache | 109,1 s (desvio 0,79; 12 execuções) |
| Tempo por imagem com 3 perguntas, com cache | cerca de 136 s |
| Pico de memória | 13.174,5 MB de RSS. Não há GPU, então não é VRAM |
| Carga do modelo | 6,56 s |

### Semana 13: alucinações (`experiments/figs/alucinacoes_configA/`)

5 categorias, 10 figuras. Principais: defeito real negado (36 respostas) e referência a contexto inexistente (15).

## Achados

1. O modelo quase não separa imagem normal de anômala em q1. Diz "Yes" em 18 de 30 anômalas e em 5 de 10 normais. Há ainda uma resposta "Answer: Yes." em imagem anômala.
2. O acerto em q1 está inflado. Metade dos 18 "Yes" em anômalas fala de blur ou reflexo, não do defeito real.
3. `contamination` é o pior grupo, com 0,100. Nenhuma resposta acertou localização ou tipo.
4. O acerto de 0,767 em `good` vem em boa parte de negar defeito por padrão, o mesmo comportamento que gera erro nas anômalas.
5. Em 14 de 40 imagens, as respostas se contradizem: "Yes" em q1 e negação em q2 ou q3.

## Confiabilidade da métrica

O FaithScore simplificado não é confiável neste domínio. Das 64 respostas erradas com escore, 45 (70%) receberam FaithScore 1,0. O verificador é o mesmo modelo que erra, então aceita "There is no defect" em imagens com defeito. Ele ainda penaliza frases sobre "facts", o que explica o menor valor em q2, e não o erro de conteúdo.

Recomendação: usar o acerto por gabarito como medida principal na comparação A vs. B, e o FaithScore como secundária, com essa limitação declarada.

## Limitações

- Uma categoria, 10 imagens por grupo, sem intervalo de confiança nos valores acima.
- O acerto por gabarito usa regras de palavras-chave. Ele pode errar respostas com outras palavras e deve ser auditado por amostra.
- As categorias de alucinação vêm de regras, ainda sem revisão humana.
- A validação da semana 10 usou fotos comuns e `longest_edge` diferente do `exp001`, não imagens do MVTec.
- A pergunta q2 pressupõe que há defeito, o que pode induzir resposta em imagens normais.
- A latência foi medida só com q1, em 4 imagens.

## Pauta para a reunião

1. Aprovar o acerto por gabarito como métrica principal e o FaithScore como secundária.
2. Validar a métrica em imagens do MVTec, com anotação manual de cerca de 30 respostas.
3. Aprovar as otimizações de inferência (`longest_edge` 1152 e cache).
4. Definir se a Configuração B roda neste note, no Mac M4 ou no Colab. O custo de A e B precisa vir da mesma máquina.
5. Decidir se q2 deve ser reformulada, sabendo que isso exige refazer a Configuração A.
6. Definir se vale testar um verificador diferente do SmolVLM.

## Decisões da reunião

1. Acerto por gabarito como métrica principal e FaithScore simplificado como secundária, a conferir na avaliação do artigo.
2. Validação manual no MVTec: a definir.
3. Otimizações aprovadas por enquanto, desde que exatamente iguais em A e B. A revisão do artigo decide se ficam.
4. A Configuração B roda neste notebook. Outras máquinas podem ser avaliadas depois.
5. Reformulação da q2: a definir.
6. Verificador: a definir; por enquanto, fica o SmolVLM.
7. PatchCore mantido por enquanto.
