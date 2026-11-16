# Resultados finais — Emenda 1 (3 backbones × 6 estratégias × 3 seeds)

Corrido de 15/11 22:35 a 16/11 08:51 na GPU (RTX 5060 Laptop, torch 2.14.1+cu130).
54 corridas, 15 pontos cada (5% → 40% do pool), L inicial = 5% **por classe**.
Bootstrap cruzado (imagens × seeds), B=400, IC 95% percentil. Tudo executado; nada estimado.

```bash
python src/al_ft.py --backbone cnn --teto --seeds 0 1 2
python src/al_ft.py --backbone vit --teto --seeds 0 1 2
python src/al_ft.py --backbone cnn --seeds 0 1 2       # ~5,7 h
python src/al_ft.py --backbone vit --seeds 0 1 2       # ~3,9 h
python src/al.py --seeds 0 1 2 --saida runs/al_porclasse   # DINOv2, ~21 min
python scripts/comparar_backbones.py                   # tabelas + figura
```

| Backbone | Teto F1 | F1 a 40% aleatória | Melhor F1 a 40% | Rótulos p/ 95% do teto (aleatória → melhor AL) | ΔALC vs aleatória (significativas) |
| --- | --- | --- | --- | --- | --- |
| CNN — EfficientNet-B0, fine-tune | 0,866 | 0,849 | 0,872 (CoreSet) | 18,3% → 11,5% (BADGE) | **5/5**: +0,018 a +0,021 |
| Transformer — ViT-S/16, fine-tune | 0,888 | 0,871 | 0,894 (CoreSet) | 13,8% → 10,4% (CoreSet/BALD) | **5/5**: +0,010 a +0,017 |
| DINOv2 ViT-S/14 congelado (no fim) | 0,831 | 0,813 | 0,831 (BALD) | 14,9% → 10,7% (entropia/margem) | **3/5**: BALD +0,018, entropia +0,015, margem +0,015; BADGE e CoreSet com IC a incluir 0 |

Tabela completa: `paper/tabelas/comparacao_backbones.md`. Figura: `paper/figuras/curvas_3_backbones.png`.
Por backbone: `paper/tabelas/{cnn,vit,dino}/` (Tabela 1, diferenças emparelhadas com F1 por classe, ECE).

**Leitura.**
1. **H1 confirmada nos 3 backbones**: há sempre estratégias que batem a aleatória com IC que exclui 0.
2. **Fine-tune dá curvas mais inclinadas** (0,75 → 0,87–0,89) do que o DINOv2 congelado (0,75 → 0,83) — o problema "muito eficiente" levantado pelo professor.
3. **O CoreSet muda de sinal com o backbone**: é dos melhores com fine-tune (as características mudam em cada ronda) e não bate a aleatória com embeddings DINOv2 fixos, como já acontecia na 1.ª ronda de resultados (abaixo).
4. **Classe rara (eflorescência)**: na CNN o ganho é significativo para CoreSet (+0,072), BALD (+0,061) e BADGE (+0,056); no ViT e no DINOv2 é positivo mas com IC a incluir 0 (H2 parcial).
5. Limitações: 3 seeds (IC das seeds aproximado); hiperparâmetros afinados numa só seed e o melhor lr ficou no limite da grelha; o teto do DINOv2 é o de `runs/baseline` (a 100% dos rótulos o L inicial não interessa).

---

# (Histórico) Primeiras corridas — 27/10/2026, só DINOv2 congelado, L "classe mais rara"


Corrido em 27/10/2026, CPU (20 núcleos lógicos, sem GPU), Python 3.11, torch 2.14.1+cpu,
timm 1.0.30. Tudo o que está aqui saiu de execução real; nada é estimado.

Reproduzir do zero:

```bash
python src/dados.py --verificar --gerar-json
python src/extrair.py --modelo dinov2_vits14      # 475 s
python src/baseline.py --experiencia ambos
python src/al.py --seeds 0 1 2                    # ~2,5 min por estratégia em paralelo
python src/avaliar.py --B 400                     # ~5 min
python src/redundancia.py
```

---

## 1. Capacidade de referência

| Experiência | Macro-F1 no teste |
| --- | --- |
| k-NN (k=20, cosseno) na validação — **sanidade do extrator** | **0,7000** (acaso 0,2551) |
| **L-só** (5% = 322 recortes) | 0,7665 |
| **Teto** (100% = 6 438 recortes), média de 3 seeds | **0,8306** (0,8296 / 0,8296 / 0,8327) |

O extrator é bom: um k-NN com 20 vizinhos já resolve 0,70 sem treinar nada, contra 0,26 de
acaso medido (rótulos de validação baralhados). O valor sai de `python src/avaliar.py --knn 20`
— k-NN multi-rótulo, cosseno, treino em `train`, medido em `val` — e é o que fica registado no
`resultados.csv`, para que o número do relatório e o número do script não possam divergir
(uma medição anterior, com outro protocolo de decisão, dava 0,7315; essa foi substituída).
E o intervalo
entre 5% e 100% dos rótulos é **estreito** — 0,77 contra 0,83. Isso não é um defeito do
desenho: é o resultado que qualquer trabalho de AL com ViT congelado tem de enfrentar. Se a
curva sobe 0,06 em 40% do orçamento, o ganho de uma estratégia tem de ser medido contra a
aleatória **com o mesmo orçamento e teste emparelhado**, não contra "o acaso".

Consequência prática para a apresentação: a frase "o AL poupa rótulos" tem de vir com
números, porque a intuição de que há muito a ganhar é falsa neste regime.

## 2. As seis estratégias (Tabela 1)

Bootstrap **cruzado** (imagens de teste × seeds), B=400. IC 95% percentil.

| Estratégia | ALC | IC 95% | Rótulos p/ 95% do teto | F1 final |
| --- | --- | --- | --- | --- |
| **BALD** | **0,8162** | [0,7801; 0,8474] | **8,7%** | 0,8290 |
| entropia | 0,8113 | [0,7738; 0,8422] | 9,1% | 0,8255 |
| margem | 0,8107 | [0,7733; 0,8417] | 9,8% | 0,8276 |
| BADGE | 0,8072 | [0,7674; 0,8398] | 9,0% | 0,8203 |
| CoreSet | 0,8002 | [0,7612; 0,8320] | 13,8% | 0,8184 |
| aleatória | 0,7945 | [0,7578; 0,8245] | 13,1% | 0,8058 |

Os IC **sobrepõem-se todos** — o que não significa "não há diferença", significa que este
teste não a consegue isolar. A decisão tem de vir da diferença emparelhada (§3).

## 3. A decisão que a H1 exige: diferença emparelhada

Mesmas imagens e seeds reamostradas para as duas estratégias em cada réplica.

| Estratégia | Δ ALC vs aleatória | IC 95% | P(>0) |
| --- | --- | --- | --- |
| **BALD** | **+0,0218** | [+0,0109; +0,0321] | 1,00 |
| **entropia** | **+0,0168** | [+0,0067; +0,0269] | 1,00 |
| **margem** | **+0,0162** | [+0,0064; +0,0267] | 1,00 |
| **BADGE** | **+0,0128** | [+0,0024; +0,0248] | 0,99 |
| CoreSet | +0,0057 | [−0,0084; +0,0183] | 0,82 |

**H1 confirmada:** quatro estratégias batem a aleatória com IC que exclui zero. O efeito é
pequeno (+0,013 a +0,022 de ALC) mas consistente — nas 400 réplicas, o BALD ganhou sempre.

**H3 refutada:** CoreSet (diversidade pura) é a única que **não** bate a aleatória. A
expectativa de que a diversidade ganharia no *cold start* não se confirmou neste dataset: com
embeddings de um ViT forte, os métodos de incerteza já cobrem a diversidade de facto. O BADGE,
híbrido, fica no meio e passa por pouco.

## 4. H2 — raridade e criticidade não são a mesma coisa

F1 por classe a ~20% dos rótulos, com a diferença emparelhada contra a aleatória:

| Classe | aleatória | BALD | Δ BALD | IC 95% do Δ |
| --- | --- | --- | --- | --- |
| Background | 0,863 | 0,887 | +0,023 | inclui 0 |
| **CorrosionStain** (crítica) | 0,695 | 0,751 | **+0,057** | **exclui 0** |
| Crack | 0,812 | 0,819 | +0,007 | inclui 0 |
| Efflorescence (mais rara) | 0,722 | 0,778 | +0,055 | inclui 0 |
| ExposedBars (crítica) | 0,872 | 0,890 | +0,020 | inclui 0 |
| Spallation | 0,809 | 0,819 | +0,009 | inclui 0 |

**H2 parcialmente confirmada, e o padrão é o que a correção do protocolo previa.** A classe
**crítica** (corrosão) ganha de forma significativa; a classe **mais rara** (eflorescência)
ganha +0,055 mas com IC a incluir zero — porque com L=5% há **27** positivos dela no conjunto
rotulado, e 27 exemplos não dão potência para nada. Foi exatamente por isso que a H2 deixou de
dizer "as classes raras" e passou a separar raridade de criticidade: se tivéssemos mantido a
formulação antiga, o resultado seria lido como falha, quando é o contrário — o método funciona
onde há sinal para medir.

## 5. H4 — calibração

ECE marginal por classe (10 bins), média das 3 seeds:

| Estratégia | ECE a 5% | ECE a 40% |
| --- | --- | --- |
| BADGE | 0,0602 | **0,0404** |
| entropia | 0,0602 | 0,0421 |
| margem | 0,0602 | 0,0428 |
| aleatória | 0,0602 | 0,0451 |
| CoreSet | 0,0602 | 0,0451 |
| BALD | 0,0613 | 0,0453 |

**H4 não se confirma:** os modelos ativos **não** ficam pior calibrados; todos melhoram com
mais rótulos, e as diferenças entre estratégias são pequenas (< 0,005 de ECE). É um resultado
negativo útil: a preocupação "o AL descalibra o modelo" não aparece aqui, e a app pode mostrar
probabilidades sem um aviso extra de calibração — mas também não há ganho a vender.

## 6. Ângulo drone — o que foi medido, não o que se supõe

`python src/redundancia.py` → `runs/redundancia.json`:

| Medida | Valor |
| --- | --- |
| Similaridade cosseno **intra-imagem** (quase-duplicados reais) | média **0,631** |
| Similaridade **entre imagens** | média 0,424 · p99 0,782 · p99,9 **0,868** |
| **τ proposto** (percentil 99,9 dos pares entre imagens) | **0,868** |
| Recortes do pool fundidos por τ | 6 438 → **4 402 grupos** (−31,6%) |
| Candidatos quase-duplicados de um rótulo já escolhido, após L=5% | **1 387 (21,5% do pool)** |

É a justificação numérica do eixo drone: **um quinto do pool é redundante** em relação ao que
já foi anotado. Num vídeo de drone essa fração é maior, porque 30 quadros por segundo do mesmo
defeito são quase-duplicados por construção. Sem o filtro, a aquisição gasta um quinto do
orçamento a re-perguntar o que já sabe — e a curva parece pior sem que o método seja pior.

O limiar não é universal (não existe um publicado para betão; a pesquisa do A1 confirma-o), é
o **deste** dataset, medido em 21 668 pares intra-imagem e 299 605 pares entre imagens.

## 7. O que **não** está feito

Lista curta e explícita, para não haver surpresa na defesa:

1. **Baseline EfficientNet afinada (PL08 §2.3)** — não corrida. Sem GPU, o fine-tune a 6 438
   imagens não cabe no tempo. O teto em uso é o da **representação congelada** (0,831), que é
   legítimo como denominador do "95% do teto", mas **não** é o baseline da disciplina.
2. **Campanha de anotação real** — a app está feita e a fila é gerada por incerteza, mas não há
   anotações humanas medidas. O custo em segundos por recorte será um piloto, não um estudo.
3. **SegCODEBRIM / segundo domínio** — não começado; continua como extensão para 03/12.
4. **3 seeds** — o IC das seeds é aproximado, como a Tabela 1 diz. Mais seeds é o caminho mais
   barato para estreitar tudo o que aqui está largo (nomeadamente a eflorescência).
5. **`PROTOCOLO.md` ainda não fechado/versionado** — e o teste já foi lido nestas corridas.
   Consequência: estas curvas são **exploratórias**; para o relatório final têm de ser
   re-corridas depois do commit do protocolo, como o próprio protocolo manda. É a limitação
   mais séria deste lote e não se disfarça.
6. **Repositório git não inicializado** (P6.1, A1) — a coluna `commit` de `paper/resultados.csv`
   está `sem-git`. Enquanto isso durar, nenhuma linha de resultados é rastreável a um código.
   Inicializar o repositório e correr de novo o que já está feito resolve isto em minutos; não foi
   feito aqui porque criar/commitar um repositório é decisão do grupo, não de um agente.

## 8. Por pessoa — o que ficou entregue

| Membro | Entregue nesta ronda | Em aberto |
| --- | --- | --- |
| **A1 Fausto** (dados) | `src/dados.py` (PL08 §2.2, determinismo confirmado) · `src/extrair.py` (7 677 × 384, retomável) · `data/splits.json` com hashes e L0 por seed · `src/redundancia.py` (τ=0,868) | Inicializar o repositório git e convidar os docentes (P6.1) · inspeção visual de 5 recortes por classe (C1.2b) · declarar a licença AGPL do Ultralytics se entrar no repo |
| **A2 Gabriela** (baseline) | `src/probe.py` (BCE, limiares por classe só na validação) · `src/baseline.py` (teto 0,8306 · L-só 0,7665) | EfficientNet afinada a 100% (P8.6) · latência no hardware alvo (D3) · confirmação com ResNet-50 para aleatória + melhor estratégia (C2.12) |
| **A3 Gonçalo** (investigador) | `src/al.py` com as 6 estratégias (incl. BALD com dropout, CoreSet k-center, BADGE multi-rótulo) · 18 corridas · tabela de diferenças emparelhadas | Fechar e versionar o `PROTOCOLO.md` (C2.6) · seminário de 17/11 (C2.10–C2.11) · falar com o orientador sobre vídeo de drone (D5) · re-verificar o BADGE com fonte externa |
| **A4 Pedro** (produto) | `src/app.py` (fila por incerteza, previsão oculta, tempo por recorte, resumo de custo) · app servida e verificada (HTTP 200) | Campanha de anotação real (nem 1 anotação existe) · `src/enviesamento.py` (composição por ronda, C2.4) · rótulos por ponte na app (D4) |
| **A5 Marco** (avaliador) | `src/avaliar.py` (ALC, rótulos-para-95%, bootstrap cruzado por imagem × seed, ECE, Tabelas 1–3, figura) | Grad-CAM (C4.2) · corrida única no teste depois do protocolo fechado (C4.3) · top-10 comentado (C3.7) |

O plano detalhado de cada um continua em [`docs/papeis/`](papeis/); esta tabela é o estado, não o substitui.

## 9. Ficheiros produzidos

| O que | Onde |
| --- | --- |
| Pipeline de dados (PL08) | [`src/dados.py`](../src/dados.py) |
| Extrator DINOv2 | [`src/extrair.py`](../src/extrair.py) |
| Teto e L-só | [`src/baseline.py`](../src/baseline.py) |
| Simulador de AL (6 estratégias) | [`src/al.py`](../src/al.py) |
| Avaliação, IC, calibração | [`src/avaliar.py`](../src/avaliar.py) |
| Redundância / τ (drone) | [`src/redundancia.py`](../src/redundancia.py) |
| Assistente de anotação | [`src/app.py`](../src/app.py) |
| Embeddings | `runs/emb_vits14.npy` (7 677 × 384) |
| Corridas | `runs/al/<estratégia>/seed<k>/` |
| Tabelas | [`paper/tabelas/`](../paper/tabelas/) |
| Figura | [`paper/figuras/curva_al.png`](../paper/figuras/curva_al.png) |
| Registo de experiências | [`paper/resultados.csv`](../paper/resultados.csv) |
