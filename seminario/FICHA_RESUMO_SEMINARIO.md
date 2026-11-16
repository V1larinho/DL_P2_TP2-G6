# Ficha-resumo de 1 página — Seminário I (Tema C)

> Fonte: PL08 §2.4. Entregar no Moodle até 15/11 (prazo seguro; a PL08 diz "domingo 16/11" mas 16/11/2026 é segunda-feira — ver REVISAO.md R19). Traga pronta para o ensaio da semana seguinte.

---

## ACTIVE LEARNING PARA DEFEITOS EM PONTES DE BETÃO (CODEBRIM) — TEMA C / AL 1

**Problema.** Anotar recortes de fissura, delaminação, corrosão, armadura exposta e eflorescência é lento, caro e desequilibrado (as patologias críticas são raras; a eflorescência é 8,4% do treino). O Active Learning gasta o orçamento de anotação onde rende mais. Pergunta central: com que fração dos rótulos uma estratégia ativa atinge 95% do desempenho do treino completo, face à amostragem aleatória — e isso vale sobretudo nas classes raras?

**Dados.** CODEBRIM (Mundt et al., CVPR 2019; Zenodo 10.5281/zenodo.2620293): 7 677 recortes utilizáveis de 1 522 imagens de origem; treino 6 438 / validação 611 / teste 628; divisão por imagem (0 imagens em duas partições); 25,9% dos recortes são multi-rótulo (29 combinações). Licença não comercial; não redistribuir. Métrica principal: Macro-F1 multi-rótulo no teste.

**Método.** Três backbones, as mesmas estratégias nos três: **CNN** EfficientNet-B0 e **Transformer** ViT-S/16 com *fine-tune* (re-treinados do pré-treino em cada ronda) e, no fim, **DINOv2** ViT-S/14 congelado + cabeça linear; BCE por classe. Estratégias de aquisição: aleatória (baseline do AL), entropia/margem multi-rótulo, BALD (MC Dropout na cabeça), CoreSet (k-center greedy em embeddings), BADGE (embeddings de gradiente + k-means++). L = 5% por classe, K = 2,5% por ronda, orçamento total 40% → 14 rondas, 3 seeds. 54 corridas (3 backbones × 6 estratégias × 3 seeds). Hiperparâmetros congelados na validação com a estratégia aleatória.

**Ideia central (figura).** Ciclo: pool não rotulado → estratégia de aquisição escolhe K recortes → anotar → treinar cabeça → adquirir novamente. A incerteza e a diversidade orientam onde anotar a seguir.

**Resultados (16/11/2026, GPU, IC 95% bootstrap imagens × seeds).** Tetos (100% rótulos): CNN 0,866 · ViT 0,888 · DINOv2 0,831. Com 40% dos rótulos a melhor estratégia iguala ou passa o teto do próprio backbone (CNN 0,872, ViT 0,894, DINOv2 0,831); a aleatória fica abaixo (0,849 / 0,871 / 0,813). Ganho de ALC sobre a aleatória com IC que exclui zero: as 5 estratégias na CNN (+0,018 a +0,021) e no ViT (+0,010 a +0,017); no DINOv2 só as de incerteza (BALD +0,018, entropia/margem +0,015). Rótulos para 95% do teto: 18% → 12% (CNN), 14% → 10% (ViT), 15% → 11% (DINOv2). CoreSet: dos melhores com *fine-tune*, inútil com embeddings congelados. Eflorescência (rara) ganha de forma significativa só na CNN (+0,06 a +0,07).

**Aplicação.** Assistente de anotação em Gradio: fila "anotar a seguir" ordenada por incerteza/diversidade, com a razão de cada escolha; previsão oculta por omissão (evitar ancoragem); re-treino da cabeça em segundos; curva de validação atualizada. Aviso: protótipo experimental, não substitui a inspeção.

**Custo.** DINOv2: embeddings uma vez (~8 min CPU) e ~1 min por corrida. Fine-tune numa RTX 5060 Laptop: ~19 min por corrida (CNN) e ~13 min (ViT); as 36 corridas com fine-tune ≈ 10 h.

**Equipa (5 membros):** A1 dados, A2 baseline, A3 investigador/porta-voz, A4 produto, A5 avaliação. Todos programam, todos escrevem, todos sabem explicar qualquer parte na defesa (07/01/2027).

**Calendário até P2-D5:** P2-D1 (proposta, 29/10) → Seminário I (17/11) → P2-D3 protótipo (03/12) → P2-D4 avaliação final (17/12) → Relatório + repositório (04/01) → Defesa individual (07/01).
