# Plano do Seminário I — 17/11/2026 (25 min + 5 min perguntas)

> Fonte: PL08 §2.4. Porta-voz: A3 (Gonçalo Vilarinho). As perguntas podem ir a qualquer membro.

## Estrutura (6 blocos, PL08 §2.4)

### Bloco 1 — Problema e motivação (3 min) · A3

- Utilizador: equipa de inspeção de infraestruturas
- Decisão: que recortes anotar a seguir; quando parar
- Porque importa: rotular é caro; patologias críticas são raras (corrosão, armadura exposta)
- Pergunta central: com que fração dos rótulos uma estratégia ativa atinge 95% do teto, face à aleatória?
- **Dizer em voz alta:** "Não é sobre fazer ML melhor — é sobre gastar o orçamento de anotação onde rende mais."

### Bloco 2 — Dataset e métricas (5 min) · A1 suporta

- CODEBRIM: 7 677 recortes de 1 522 imagens, 5 classes + fundo, multi-rótulo (25,9%)
- Contagens por classe: fissura 2 507 · delaminação 1 898 · corrosão 1 559 · armadura exposta 1 507 · eflorescência 833 (a mais rara)
- Divisão por imagem (não por recorte): 0 imagens em duas partições
- Métrica: Macro-F1 multi-rótulo no teste; resumos: ALC e rótulos para 95% do teto com IC 95%
- **Mostrar:** contagens por classe no README; a distinção raridade ≠ criticidade

### Bloco 3 — Método (7 min) · A3 apresenta; A2, A5 prontos a responder

- Três backbones, **as mesmas 6 estratégias nos três** (pedido do professor):
  - CNN: EfficientNet-B0 (ImageNet-1k), *fine-tune*
  - Transformer: ViT-S/16 (ImageNet-21k→1k), *fine-tune*
  - DINOv2 ViT-S/14 congelado + cabeça linear — **no fim**, é a referência "muito eficiente"
- No *fine-tune*, em cada ronda: re-treinar do pré-treino com o L atual → extrair h(x) → a mesma cabeça e o mesmo código de aquisição do caso congelado
- Estratégias: aleatória (baseline) · entropia/margem · BALD (MC Dropout T=10) · CoreSet (k-center greedy) · BADGE (gradiente + k-means++)
- Protocolo: L = 5% **por classe**, K=2,5%, 14 rondas, 3 seeds → 54 corridas; hiperparâmetros afinados na validação com a aleatória
- **Mostrar:** `paper/figuras/curvas_3_backbones.png`

### Bloco 4 — Resultados (4 min) · A5 suporta

- Tetos: CNN 0,866 · ViT 0,888 · DINOv2 0,831
- H1 confirmada nos 3: CNN 5/5 estratégias batem a aleatória (+0,018 a +0,021 ALC), ViT 5/5 (+0,010 a +0,017), DINOv2 3/5 (BALD, entropia, margem)
- Rótulos para 95% do teto: aleatória 18% / 14% / 15% → melhor AL 12% / 10% / 11%
- CoreSet muda de sinal: bom com *fine-tune*, igual à aleatória com DINOv2 congelado
- H2 parcial: eflorescência ganha de forma significativa só na CNN
- **Dizer:** o *fine-tune* torna a curva mais inclinada e é aí que o AL tem espaço; com DINOv2 já se começa perto do teto

### Bloco 5 — Aplicação (3 min) · A4 suporta

- Gradio: fila "anotar a seguir" com razão por recorte (incerteza/diversidade/classe rara provável)
- Previsão oculta por omissão (evitar ancoragem — Rosbach et al. 2026, 7% de reversão)
- Re-treino da cabeça em segundos; curva atualizada em tempo real
- Aviso: "Protótipo experimental; não substitui a inspeção"
- **Dizer:** o que nos distingue de um classificador vulgar é a fila com razão + re-treino em loop

### Bloco 6 — Conclusão e próximos passos (3 min) · A3

- O AL funciona nos 3 backbones; o efeito é consistente (+0,01 a +0,02 ALC) e poupa cerca de ⅓ dos rótulos para chegar a 95% do teto
- Vale a pena quando: classes raras são críticas; orçamento é muito limitado (<10%)
- Próximos passos: mais seeds; campanha de anotação real; SegCODEBRIM
- **Fechar com:** "Se a aleatória ganhar, o resultado é esse — com análise do porquê. Já acontece na literatura."

---

## Notas de ensaio

- Cronometrar e cortar até 23 min (sobra margem para perguntas antecipadas)
- Ensaio na PL09 (semana 9, 11–12/11)
- Perguntas prováveis: diferença emparelhada vs IC na Tabela 1; porque L estratificado; porque CoreSet falhou; BADGE multi-rótulo vs single-label original

---

## Questões preparadas (qualquer membro deve saber responder)

1. Porque é que a comparação justa para AL é a aleatória com o mesmo orçamento, e não "uma CNN simples"?
2. O bootstrap é por recorte ou por imagem, e por quê?
3. Se o BALD ganha apenas +0,022 de ALC, isso justifica a complexidade adicional?
4. Porque é que o CoreSet funciona com *fine-tune* mas falha com embeddings DINOv2 congelados?
6. Porque re-treinar do pré-treino em cada ronda em vez de continuar o treino anterior?
7. Como se comparam backbones com tetos diferentes (ALC normalizado pelo teto do próprio backbone)?
5. Como é que a ECE multi-rótulo é calculada aqui (marginal por classe)?
