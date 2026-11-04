---
marp: true
title: P2-D1 — Active Learning para defeitos em pontes de betão (CODEBRIM)
paginate: true
---

<!--
PITCH DO P2-D1 — 5 minutos, 5 diapositivos.
Porta-voz: A3 (Gonçalo Vilarinho). Regra de ensaio: cronometrar e cortar até caber em 4 min 30 s,
para sobrar margem. Cada bloco da grelha da PL07 §2.1 vale 20% da nota.
Diapositivo 1 Problema · 2 Dados · 3 Métodos · 4 Aplicação · 5 Plano.
-->

# Active Learning para defeitos em pontes de betão
### Quantos rótulos são mesmo precisos? — Tema C (AL 1)

**Grupo 6 (TP2)** · Fausto Cananga · Gabriela Cardoso · Gonçalo Vilarinho · Pedro Ribeiro · Marco Maceda

Deep Learning Aplicado · 2026/2027 · **P2-D1** · 29/10/2026

---

## 1. Problema
### (utilizador · decisão apoiada · porque importa)

- **Utilizador:** equipa de inspeção de infraestruturas que está a treinar um classificador de
  defeitos de pontes de betão a partir de fotografias de campo.
- **Decisão:** *que recortes anotar a seguir* — e *quando parar de anotar*. Hoje anota-se ao acaso.
- **Porque importa:** rotular é lento e caro, e as patologias **críticas são raras** (armadura
  exposta, corrosão). É aí que o orçamento se perde.

> **Pergunta central:** com que **fração dos rótulos** uma estratégia ativa atinge **95% do
> desempenho do treino completo**, face à amostragem aleatória — e isso vale sobretudo nas
> **classes raras**?

*(nota: 45 s — não explicar AL, dizer o problema)*

---

## 2. Dados
### (origem · licença · dimensão · unidade de agrupamento · divisão)

| | |
| --- | --- |
| Dataset | **CODEBRIM** (Mundt et al., CVPR 2019 · AEROBI · Zenodo `10.5281/zenodo.2620293`) |
| Dimensão | **1 590 imagens** · **5 354 recortes** multi-rótulo · **5 classes + fundo** |
| Classes | fissura · delaminação/`spalling` · eflorescência · **armadura exposta** · **corrosão** |
| Licença | **"Other (Non-Commercial)"** — uso académico, citação obrigatória, **não redistribuímos** |
| Unidade de agrupamento | **imagem de origem** — recortes da mesma imagem na mesma partição |
| Divisão | oficial (treino/validação/teste); **teste congelado**; pool = treino; validação só para hiperparâmetros |

**Correção face à primeira proposta:** é **classificação de recortes**, não deteção — `mAP@0,5:0,95`
**não** se aplica. A métrica passa a **Macro-F1 multi-rótulo vs fração de rótulos**.

*(nota: 60 s — se alguém perguntar pelas contagens por classe: "estão a ser contadas no dia 1 e
publicadas no README")*

---

## 3. Métodos
### (baseline lecionada · método avançado · referência)

**Baseline (a única comparação justa para AL):** **amostragem aleatória**, com o mesmo `L`, o mesmo
`K` e o mesmo orçamento. Dá a curva *passiva*.
**Teto:** modelo com **100% dos rótulos** — o desempenho a atingir.

**Método avançado (não lecionado):** estratégias de aquisição sobre **características congeladas de
DINOv2** (`ViT-S/14`, embedding CLS normalizado) com uma **cabeça linear**:

| Estratégia | Ideia |
| --- | --- |
| Entropia / margem multi-rótulo | o que o modelo **não** sabe |
| **BALD** (MC Dropout na cabeça) | o que reduz a **incerteza do modelo** |
| **CoreSet** (k-center greedy) | cobrir o pool — **diversidade** |
| **BADGE** (gradiente + k-means++) | diversidade no **espaço dos gradientes** |

Referências: Sener & Savarese 2018 · Ash et al. 2020 · Gal et al. 2017 · Settles 2009 · Oquab et al. 2023.
Confirmamos o resultado principal com uma **CNN afinada**.

*(nota: 90 s — este é o bloco com mais peso; desenhar o ciclo pool → modelo → aquisição → oráculo)*

---

## 4. Aplicação
### (mockup: entradas · saídas · avisos)

**Assistente de anotação** (Gradio). **Entrada:** lote de recortes não rotulados + escolha da
estratégia + orçamento `K`.

**Saídas:**
1. defeitos previstos por recorte, com **confiança calibrada**;
2. **fila "anotar a seguir"** com a **razão** de cada escolha (incerteza / diversidade / classe rara provável);
3. anotação multi-rótulo na própria app → **re-treino da cabeça em segundos** → curva atualizada;
4. composição do conjunto rotulado (para o viés que o AL cria ficar visível).

**Avisos:** *"Protótipo experimental; previsões de um classificador de recortes treinado no CODEBRIM
— não substitui a inspeção"*; recortes **fora da distribuição** são sinalizados, não classificados.

*[inserir aqui a imagem do mockup de `docs/P2-D1.md` §4]*

*(nota: 45 s — apontar a fila e a razão, que é o que nos distingue de um classificador vulgar)*

---

## 5. Plano
### (calendário · papéis · estimativa de treino)

| Marco | Data | Quem |
| --- | --- | --- |
| `src/dados.py` + verificação de fuga · baseline | 05/11 | A1, A2 |
| Entropia/BALD/CoreSet/BADGE · `PROTOCOLO.md` fechado | 12/11 | A3, A5 |
| **Seminário I** (25 min) + ficha de 1 página | 17/11 | A3 |
| Assistente de anotação · **P2-D3** | 03/12 | A1, A4 |
| ECE, Grad-CAM, teste oficial · **P2-D4** | 17/12 | A5, A4 |
| Relatório + repo · **P2-D5** / defesa | 04/01 · 05–07/01 | todos |

**Papéis:** A1 dados/protocolo/integração · A2 classificadores e teto · A3 estratégias (porta-voz) ·
A4 classes raras e enviesamento · A5 avaliação, calibração e XAI.

**Custo:** extração DINOv2 **uma vez**; rondas com cabeça linear = `5 estratégias × 3 seeds × 14
rondas` e cabem num **Colab gratuito**. O gargalo é a CNN afinada → só para **2 estratégias × 3 seeds**,
com medição de tempo/época antes de arrancar. Protocolo fixado: `L=5%`, `K=2,5%`, orçamento `40%`
(**14 rondas**), 3 seeds, hiperparâmetros congelados na validação com a aleatória.

*(nota: 60 s — fechar com o risco principal: se a aleatória ganhar, o resultado é esse, com a análise
do porquê)*
