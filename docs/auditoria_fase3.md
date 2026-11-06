# Auditoria da Fase 3

Data: 05/11/2026. Revisão feita pelo Claude a pedido do aluno. Não substitui a anotação humana: o aluno deve confirmar.

## exp001

49 afirmações conferidas nas 5 imagens. 1 marcação mudou: "The background is blurred" passou de falsa para verdadeira, porque o fundo visível nos cantos está desfocado. Duas especulações receberam nota de conteúdo analítico, e 1 afirmação ambígua foi mantida com nota. Status: `assistant_reviewed_pending_student_confirmation`.

Efeito: verificáveis passaram de 36 para 37 de 49 (0,755). No `exp004`, recalculado com os P(yes) já salvos: acurácia 0,878, kappa 0,602, 6 de 12 afirmações falsas aceitas, Pearson 0,419 (manuais) e 0,957 (automáticas).

## Acerto por gabarito

Foram auditadas 15 respostas sorteadas (semente 42, 5 por pergunta) e todos os 55 padrões distintos de resposta. As posições de q2 foram conferidas com a máscara.

- 1 erro de regra corrigido: "Answer: Yes." ficava indefinida e agora conta como "Yes".
- Nas 15 sorteadas, a nota final bate com a avaliação de presença, posição e tipo.
- Ressalva mantida: em q1, 2 das 15 acertam a presença citando blur, não o defeito. No geral, são 9 de 18 "Yes" em anômalas.
- q2 é permissiva: "middle" conta quando a máscara toca o centro. Em quebras grandes isso favorece o acerto.

Efeito: acerto geral de 0,448 para 0,453 (n=117); q1 de 0,590 para 0,600; `broken_small` de 0,407 para 0,429.

## Galeria

As 10 figuras foram conferidas. Nenhuma categoria mudou.

- `bottle_broken_large_003`: a imagem tem mesmo uma área desfocada. "blur" é verdadeiro, mas não descreve o defeito.
- "Defeito inventado": os reflexos existem. O erro é tratá-los como defeito.
- 1 resposta errada segue sem categoria: `bottle_broken_small_009`, q2, "in the middle", com defeito embaixo. É erro de localização, sem outro caso igual.
