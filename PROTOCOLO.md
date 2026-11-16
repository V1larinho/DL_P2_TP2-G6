# PROTOCOLO.md — protocolo experimental (versão final)

**Fecho e versão:** fechado na PL09, ANTES de qualquer avaliação no conjunto de teste.
Depois do commit, o teste só é lido para produzir as curvas finais — nunca para decidir nada.
Qualquer alteração posterior ao protocolo ganha uma secção "Emendas" no fim, com data e motivo.

Base: Protocolo do Projeto P2 — Tema C (AL 1), §6. Dono: **A1** (Fausto Cananga).
Versão: **final (fechado)**.

---

## 1. Hipóteses (a testar, não a confirmar)

| ID | Hipótese | Como se decide |
|---|---|---|
| **H1** | Pelo menos uma estratégia ativa atinge **95% do Macro-F1 do teto** com **menos rótulos** do que a aleatória | IC 95% (3 seeds) do número de rótulos necessários, **não sobreposto**; a fração é reportada, não imposta |
| **H2** | O ganho **concentra-se nas classes escassas e nas críticas** — e no CODEBRIM **não são as mesmas**: a mais rara é a **eflorescência** (833 recortes totais, 543 no treino); as críticas são **armadura exposta** (1 507) e **corrosão** (1 559) | F1 **por classe** a 20% de rótulos: superior com AL do que com aleatória na eflorescência **e** no par crítico; **IC da diferença emparelhada por seed a excluir 0**. Se só um dos dois se confirmar, isso é o resultado |
| **H3** | Estratégias de **diversidade/híbridas** (CoreSet, BADGE) superam a **incerteza pura** nas primeiras rondas (*cold start*), e a diferença desaparece com o orçamento | comparação por ronda com IC por seed |
| **H4** | (exploratória) O modelo treinado com amostras ativas está **pior calibrado** (ECE maior) do que o aleatório, para o mesmo orçamento | ECE por ronda, antes e depois de temperature scaling |

---

## 2. Desenho da simulação

| Campo | Preenchimento |
|---|---|
| Unidade de agrupamento e divisão | **Imagem de origem** (é a unidade do paper, não a ponte). Divisão **oficial verificada**: treino **6 438** · validação **611** · teste **628** recortes utilizáveis (**52 excluídos** por vetor todo a zero), **0** imagens de origem em duas partições (`scripts/contar_codebrim.py`). Teste oficial **congelado**; **pool = treino oficial (6 438)**; validação só para seleção de hiperparâmetros. **Limitação declarada:** com 30 pontes, imagens da mesma ponte podem cair em partições diferentes → **não** medimos generalização para pontes novas |
| `L` (inicial) | **5%** do pool, **estratificado** — ver §3 (regra de estratificação fixa); **igual para todas as estratégias** na mesma seed |
| `K` (por ronda) | **2,5%** do pool |
| Orçamento total | **40%** do pool → **14 rondas** (`(40 − 5) / 2,5`) |
| Seeds | **3** (fixam `L` **e** a inicialização); as mesmas para todas as estratégias |
| Re-treino | **de raiz em cada ronda** para a cabeça; *warm start* só para a ResNet-50, declarado como tal |
| Hiperparâmetros | escolhidos **uma vez** na validação, **com a estratégia aleatória**, e congelados; **nunca re-selecionados por estratégia** |
| Registo | `splits.json` + listas adquiridas por ronda com **hash**; uma linha por ronda em `paper/resultados.csv` |
| Rondas | 14 · estratégias 5 (aleatória, entropia/margem, BALD, CoreSet, BADGE) · seeds 3 → **210 rondas** com cabeça linear |

---

## 3. Definições de método (fixadas)

| Item | Definição fixada |
|---|---|
| Extrator | DINOv2 **ViT-S/14** (confirmação com ViT-B/14), embedding **CLS**, **normalizado L2**; pesos congelados, sem fine-tuning |
| Cabeça | **linear** (confirmação: MLP de 1 camada escondida); `BCEWithLogitsLoss`; normalização dos embeddings como pré-processamento |
| `L` estratificado (multi-rótulo) | Estrato de cada recorte = **classe positiva mais rara** que ele tem (por frequência de classe no pool). Garante pelo menos 1 exemplo por classe presente em `L`; nunca mais do que o estrato tem. Com `L` = 5% (322 recortes) e eflorescência a 8,4% do pool, `L` fica com **~27 positivos** de eflorescência — número que decide a potência de H2 |
| Entropia multi-rótulo | Média das entropias **binárias** por classe: `-1/C · Σ_c [p_c log p_c + (1-p_c) log(1-p_c)]` |
| Margem multi-rótulo | Média de `\|p_c − 0,5\|` por classe (menor = mais incerto). Não usar a diferença entre as duas maiores probabilidades: com rótulos independentes essa noção não tem o significado que tem em classificação exclusiva |
| BALD | Cabeça com **dropout** (p = 0.2), `T = 10` passagens em modo treino com seed própria; `BALD = H(média das previsões) − média das entropias`. Sem dropout na cabeça o BALD **degrada para entropia** — verificado que os rankings de BALD e entropia diferem |
| CoreSet | **k-center greedy** sobre os embeddings **normalizados** (distância euclidiana ≡ cosseno) |
| BADGE | Embeddings de gradiente da **última camada** (BCE) → vetor de dimensão `C` por amostra; **k-means++** sobre essas `K` sementes |
| Thresholds | Por classe, escolhidos na **validação**; **congelados** com a estratégia aleatória e usados em todas |
| Calibração | **Temperature scaling** ajustado na validação |
| ALC | Área sob a curva de Macro-F1 vs fração de rótulos; usar `np.trapezoid` (não `np.trapz`, descontinuado); guarda contra intervalo nulo |
| Rótulos para 95% do teto | `NaN` se o alvo nunca é atingido; reportado como `NaN`, nunca como 0 |

---

## 4. Baseline(s), métricas e comparações

| Campo | Preenchimento |
|---|---|
| Métrica **principal** | **Macro-F1 multi-rótulo no teste**, em função da **fração de rótulos**; resumos: **ALC** (área sob a curva) e **rótulos necessários para 95% do teto** |
| Intervalos de confiança | 95%, por **bootstrap de imagens de teste** (reamostrar **imagens**, não recortes — respeitar o agrupamento) **e** por seed |
| Métricas secundárias | AP multi-rótulo; F1 por classe (raras); **ECE**; composição do conjunto rotulado por ronda; tempo por ronda; [extensão] Dice no SegCODEBRIM |
| Condições a comparar | **Teto** (100%) · **L-só** · aleatória · entropia/margem · BALD · CoreSet · BADGE — cabeça linear sobre DINOv2, 3 seeds. Confirmação com **CNN afinada** apenas para **aleatória e a melhor estratégia** (3 seeds) |
| Baselines do tema (PL08) | CNN pré-treinada **EfficientNet** com fine-tuning a 100% dos rótulos = baseline do projeto/teto; a comparação **própria do AL** é a **aleatória com o mesmo orçamento** |
| Conjunto de teste | Oficial, congelado; avaliado em todas as rondas, **lido só depois do commit** deste ficheiro; previsões e listas por ronda guardadas em `runs/`. **O teste oficial é balanceado por desenho** (150 exemplos de defeito por classe, 150 na validação): Macro-F1 é comparável à literatura, **não** reflete o prior de implantação — reportar sempre F1 por classe com IC e dizer isto no relatório |
| Critério de sucesso | **H1–H3 respondidas com IC.** Se nenhuma estratégia superar a aleatória, **o resultado é esse** — com análise do porquê (acontece na literatura) |
| Análises previstas | IC por bootstrap de imagens e por seed; diferença **emparelhada** entre estratégias na mesma seed e ronda; por classe; composição do conjunto rotulado; recortes mais adquiridos; curva de calibração por ronda |

---

## 5. Verificações obrigatórias (antes de gastar orçamento)

1. **V1** Para cada seed, `L` é **idêntico** em todas as estratégias, e o teste **nunca** aparece em `L` nem em `U` (conferir por **hash**).
2. **V2** Com `K ≥ |pool|`, **todas** as estratégias convergem para o teto (sanidade do simulador).
3. **V3** A estratégia aleatória com cabeça linear **reproduz-se** entre execuções com a mesma seed.
4. **V6** BADGE com `K = |pool|` seleciona **tudo**; os escolhidos são **diversos** no t-SNE; mesma seed ⇒ mesmo resultado.
5. **V5** `python src/dados.py --verificar` corre sem erro e imprime as contagens.
6. Sanidade do extrator: **k-NN (k = 20)** na validação com Macro-F1 **acima do acaso**.

---

## 6. Fuga de dados, métricas inadequadas e comparações injustas (revisão obrigatória)

Lista de riscos que este protocolo fecha antes de correr — corresponde ao pedido obrigatório "Com IA" da semana 9 (`IA-5` em `PENDENTES.md`):

| Risco | Decisão |
|---|---|
| Recortes da mesma imagem em partições diferentes | Proibido; `splits.csv` + `verificar()` por **grupo = imagem** |
| Bootstrap por **recorte** em vez de por **imagem** | Proibido: os recortes de uma imagem não são independentes; reamostrar imagens |
| Hiperparâmetros afinados **por estratégia** | Proibido: congelados na validação com a aleatória (§2) |
| Teste lido durante as rondas para orientar decisões | Proibido: só depois do commit deste ficheiro (§4) |
| Comparar estratégias com `L` ou orçamentos diferentes | Proibido: `L`, `K` e seeds iguais em tudo |
| Thresholds escolhidos com dados de teste | Proibido: thresholds só na validação |
| Reportar só a melhor seed | Proibido: 3 seeds declaradas, média ± dp + IC |
| Normalizar embeddings só em parte do pipeline | Proibido: normalização **única**, registada em `runs/` |
| Confundir "caixas anotadas" com "recortes" | Contagens do README têm de dizer explicitamente qual é qual |

---

## 7. Configuração por experiência

Um YAML por corrida em `configs/` — ex.: [`configs/al_vits14_aleatoria_s0.yaml`](configs/al_vits14_aleatoria_s0.yaml):

```yaml
experiencia: al_aleatoria_seed0
classificador: cabeca_linear          # cabeca_linear | mlp | resnet50_finetune
extrator: dinov2_vits14               # pesos congelados, embedding CLS normalizado
estrategia: aleatoria                 # aleatoria | entropia | margem | bald | coreset | badge
seed: 0
L: 0.05                               # fração do pool
K: 0.025                              # fração do pool por ronda
rondas: 14
orcamento: 0.40
dados:
  splits: data/splits.json            # com hash
  teste: congelado
hiperparametros:                      # congelados na validação com a aleatória
  epocas: 200
  lr: 0.01
  weight_decay: 1.0e-4
  threshold_por_classe: runs/thresholds.json
```

---

## 8. Emendas

### Emenda 1 (nov/2026) — três backbones e `L` estratificado por classe

**Motivo (feedback do docente):** com DINOv2 congelado a cabeça linear atinge F1 ≈ 0,75 logo com 5% dos rótulos; a curva é quase plana e o ganho marginal do AL é pequeno ("muito eficiente"). Pedido: comparar as mesmas estratégias de AL em **três backbones — CNN, Transformer e DINOv2 (este no fim)** — e estratificar `L` **por classe**.

| Campo | Antes | Depois |
|---|---|---|
| Backbone | DINOv2 ViT-S/14 congelado | fator experimental com 3 níveis: **CNN** EfficientNet-B0 (ImageNet-1k, *fine-tune*), **Transformer** ViT-S/16 (ImageNet-21k→1k, *fine-tune*), **DINOv2** ViT-S/14 congelado + cabeça linear (inalterado, apresentado em último como referência auto-supervisionada) |
| `L` inicial | 5% do pool, estrato = classe positiva mais rara | **5% dos positivos de cada classe** (`dados.selecionar_L_por_classe`): para cada classe sorteiam-se ⌈0,05·n_pos⌉ recortes que a têm; os já escolhidos por outra classe contam para a quota. \|L\| é variável (≈ 312–319 com as seeds 0–2, ≈ 5% do pool); cada classe fica com ≥ 5% dos seus positivos |
| Re-treino (CNN/ViT) | — | **do pré-treino em cada ronda** (sem *warm start*), **15 épocas**, AdamW + OneCycle, lr backbone **1e-3 (CNN) / 1e-4 (ViT)**, lr cabeça 1e-3, wd 1e-4, batch 32, 224 px, *flips* H/V, AMP em GPU (`backbones.HIPER_FT`). Escolhidos por grelha na **validação** com L aleatório (L inicial 5% por classe e L=20%; seed 0; `scripts/afinar_ft.py`, `runs/afinacao_ft/`): CNN lr ∈ {1e-4, 3e-4, 1e-3}, ViT lr ∈ {2e-5, 5e-5, 1e-4}, épocas ∈ {8, 15}. F1 val do escolhido: CNN 0,751 → 0,841, ViT 0,765 → 0,859 (5% → 20%). Nota: o melhor lr ficou no limite superior da grelha nos dois casos, com margem ≤ 0,01 sobre o seguinte |
| Estratégias | 6 | as **mesmas 6 funções** (`al.pontuar`/`al.escolher`) nos 3 backbones; em CNN/ViT operam sobre h(x) penúltimo do modelo afinado nessa ronda e a sua cabeça (BALD: dropout 0,2 na cabeça) |
| Teto | 100% do pool, cabeça linear | 100% do pool **por backbone** (`al_ft.py --teto`), para normalizar o ALC entre backbones |
| Comparação entre backbones | — | ALC normalizado pelo teto do próprio backbone e ganho de cada estratégia sobre a aleatória **dentro** do backbone; nunca F1 absoluto entre backbones como prova de AL |
| Corridas | 6 × 3 seeds | 3 backbones × 6 estratégias × 3 seeds = **54 corridas** × 15 pontos |

Hiperparâmetros de *fine-tune* fixados na validação com a estratégia aleatória e congelados, como em §2. Resultados das corridas DINOv2 anteriores à emenda (estrato "mais rara") ficam como estão; as novas usam `--estratificacao por_classe` (por defeito) e são registadas em separado.
