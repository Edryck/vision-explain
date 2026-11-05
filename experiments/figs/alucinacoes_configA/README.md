# Galeria de alucinações: Configuração A

Status: rascunho, pendente de revisão humana. As categorias vêm de regras em `src/analyze_hallucinations.py`. Contorno vermelho: máscara de ground truth. Dados de cada figura: `gallery.json`.

A seleção não usou o FaithScore, porque ele aceitou a maioria das respostas erradas. Usou o acerto por gabarito e a categoria, com as 2 primeiras respostas distintas de cada uma.

| Categoria | Respostas | Exemplo | Comentário |
|---|---|---|---|
| Defeito real negado | 36 | "There is no defect in the image." | Erro mais frequente. A quebra é visível, mas o modelo nega |
| Referência a contexto inexistente | 15 | "There is no defect mentioned in the facts provided" | Cita "facts" que não existem no prompt. Parece vício de formato de VQA |
| Objeto ou tipo errado | 8 | "It is a lens." / "It is a black hole." | A garrafa vista de cima lembra uma lente |
| Defeito inventado em imagem normal | 7 | "Yes, there is a reflection of light" | O reflexo existe, mas não é defeito. O erro é chamá-lo de defeito |
| Defeito confundido com qualidade da foto | 5 | "Yes, there is a blur in this image." | Trata defeito como problema de foco |

Contagens por resposta. Uma resposta entra em só uma categoria, na ordem de prioridade do script. A categoria "contexto inexistente" inclui respostas em imagens normais, em que negar o defeito está certo pelo gabarito, mas a citação de "facts" continua sendo alucinação.

Inconsistência entre perguntas: em 14 de 40 imagens, o modelo diz "Yes" em q1 e nega o defeito em q2 ou q3.
