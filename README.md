# DL_P2_TP2-G6 — Active Learning para defeitos em pontes de betão (CODEBRIM)

Projeto P2 da UC **Deep Learning Aplicado** (2026/2027) — **Tema C / AL 1**, Grupo **TP2-G6**.
Pergunta central: **com que fração dos rótulos uma estratégia ativa atinge 95% do desempenho do
treino completo, face à amostragem aleatória — e isso vale sobretudo nas classes raras?**

> Repositório oficial do **Grupo 6 (TP2-G6)**, privado, com os docentes como colaboradores.

---

## 1. Problema

Quem usa a aplicação é a **equipa de inspeção de infraestruturas** que vai anotar imagens de pontes
para treinar um classificador de defeitos. A decisão apoiada é operacional: **que recortes anotar a
seguir e quando parar de anotar** — e, no uso corrente, que recortes rever por serem incertos.

Anotar milhares de recortes com fissuras, delaminações, eflorescências, armadura exposta e corrosão é
lento, caro e desequilibrado: as patologias críticas são raras. O Active Learning (AL) gasta o
orçamento de anotação onde rende mais. O projeto mede **quanto** rende, com intervalos de confiança.

**Evolução pedida pelo orientador:** o mesmo problema com a inspeção feita por **drone** em vez de
pessoas. Não muda a pergunta nem a métrica, mas muda o desenho: o pool passa a vídeo (com
quase-duplicados), a unidade de agrupamento sobe a *(ponte, voo)*, o modelo tem de correr a bordo e a
incerteza serve também para decidir onde refotografar — ver [`docs/DRONE.md`](docs/DRONE.md).

## 2. Dados

| Campo | Valor |
| --- | --- |
| Dataset | **CODEBRIM** — `CODEBRIM_classification_dataset` (recortes já divididos em treino/validação/teste) |
| Origem | Mundt et al., CVPR 2019 · projeto AEROBI · Zenodo [`10.5281/zenodo.2620293`](https://doi.org/10.5281/zenodo.2620293) |
| Escala (**medida**) | **7 729 recortes** em ficheiro → **7 677 utilizáveis**: treino **6 438** · validação **611** · teste **628**, de **1 522 imagens de origem**. **52 recortes têm todas as classes a zero** e ficam de fora (ver `R25`) |
| Classes (**recortes**, treino/val/teste) | fissura **2 507** (2 208/149/150) · delaminação **1 898** (1 608/140/150) · corrosão **1 559** (1 263/146/150) · armadura exposta **1 507** (1 215/142/150) · **eflorescência 833** (543/141/149) · fundo 2 485 (2 185/150/150) |
| Multi-rótulo (**medido**) | **1 990 recortes (25,9%)** têm **≥2 defeitos**, em **29 combinações** distintas (a mais frequente: `Spallation|ExposedBars|CorrosionStain`, 789) — é esta a dificuldade real da tarefa |
| Divisão oficial (**verificada**) | Validação e teste estão **balanceados por desenho** (~150 por classe); o **treino não está** (eflorescência = 8,4% do treino). Agrupamento por **imagem**: verificado por script, **0 imagens em duas partições** |
| Ficheiro a usar | `CODEBRIM_classification_dataset.zip` (**7,9 GB**, md5 `c1612d9674e2e628e72e7f5817c40130`) — **não** usar `..._classification_balanced_dataset` (replica imagens para balancear e destrói a composição de `L`) |
| Tipo de anotação | **classificação de recortes multi-rótulo**, *não* deteção — mAP@0,5:0,95 **não se aplica** |
| Licença | **"Other (Non-Commercial)"** — uso académico, **citar Mundt et al. (2019)**; **não redistribuir** os dados (por isso `data/` está fora do git) |
| Unidade de agrupamento | **imagem de origem** — recortes da mesma imagem têm de ficar na mesma partição |
| Extensão opcional | **SegCODEBRIM** (Jaziri et al., WACV 2024, licença MIT) — máscaras de fissuras, AL para segmentação |

Tudo o que está na tabela acima foi **medido no dataset** com
[`scripts/contar_codebrim.py`](scripts/contar_codebrim.py) (zip verificado por `md5`, etiquetas lidas
dos XML de metadados, fuga por imagem verificada por grupo):

```bash
python scripts/contar_codebrim.py --listar        # estrutura + amostra com etiquetas
python scripts/contar_codebrim.py --contar --splits data/splits.csv
# -> 7 677 utilizáveis · 52 sem rótulo excluídos · 0 imagens em duas partições · exit 0
```

A comparação com SegCODEBRIM, dacl10k e as alternativas só-de-fissura — e a razão pela qual **não se
troca de dataset** (é o único multi-rótulo, desbalanceado e obrigatório pelo protocolo) — está em
[`docs/DATASETS.md`](docs/DATASETS.md).

### Citação obrigatória

```bibtex
@inproceedings{mundt2019codebrim,
  title     = {Meta-Learning Convolutional Neural Architectures for Multi-Target Concrete Defect Classification with the {CODEBRIM} Dataset},
  author    = {Mundt, Martin and Majumder, Sagnik and Murali, Sathyanarayana and Panetsos, Panagiotis and Ramesh, Vishal},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

## 3. Método

- **Três backbones, as mesmas estratégias nos três** (Emenda 1 do [`PROTOCOLO.md`](PROTOCOLO.md)):
  - **CNN** — EfficientNet-B0 (ImageNet-1k), *fine-tune* completo, re-treinada do pré-treino em cada ronda;
  - **Transformer** — ViT-S/16 (ImageNet-21k→1k), *fine-tune* completo, idem;
  - **DINOv2** `ViT-S/14` **congelado** (embeddings CLS normalizados) + cabeça linear — apresentado **no fim**,
    como a referência "muito eficiente".
- **Classificador:** a mesma cabeça linear (`probe.Cabeca`, BCE por classe) nos três; no *fine-tune* as
  estratégias veem as características penúltimas h(x) do modelo afinado, por isso o código de aquisição é o mesmo.
- **Três coisas que não se confundem** (ver `docs/REVISAO.md` R2): *baseline do projeto* (PL08 §2.3) =
  CNN pré-treinada (EfficientNet) afinada a **100% dos rótulos**; *teto* = esse regime com a cabeça
  principal; *baseline do AL* = a **aleatória** com o mesmo orçamento — é a única comparação justa.
- **Estratégias de aquisição (método não lecionado):** aleatória (baseline), incerteza multi-rótulo
  (entropia média / margem), **BALD** (MC Dropout na cabeça), **CoreSet** (k-center greedy em
  embeddings), **BADGE** (embeddings de gradiente + k-means++).
- **Métrica principal:** Macro-F1 multi-rótulo no teste **vs fração de rótulos usada**; resumos:
  **ALC** e **rótulos necessários para 95% do teto**. IC 95% por bootstrap de imagens de teste e por seed.
- **Transversal (família D):** ECE do modelo ativo vs aleatório, composição do conjunto rotulado por
  ronda, Grad-CAM nos recortes adquiridos.

Parâmetros do protocolo: `L = 5%` **por classe** (cada classe recebe ≥ 5% dos seus positivos; 3 seeds),
`K = 2,5%` por ronda, orçamento total `40%` do pool → **14 rondas**, 3 backbones × 6 estratégias × 3 seeds = **54 corridas**, hiperparâmetros congelados após seleção na validação
com a estratégia aleatória, re-treino de raiz em cada ronda. Detalhe em [`PROTOCOLO.md`](PROTOCOLO.md).

## 4. Aplicação — assistente de anotação

`src/app.py` (Gradio): recebe recortes não rotulados, prevê defeitos com confiança **calibrada**,
ordena a fila "anotar a seguir" **com a razão** (incerteza/diversidade), permite anotar e re-treinar a
cabeça em segundos, e mostra a curva em validação. Mockup em [`docs/P2-D1.md`](docs/P2-D1.md).
Aviso obrigatório na interface: *"Protótipo experimental; previsões de um classificador de recortes
treinado no CODEBRIM — não substitui a inspeção"*.

## 5. Papéis e responsabilidades

Os 5 papéis do Protocolo do Tema C, atribuídos pelo cargo de cada membro:

| Papel (protocolo) | Membro | Nº | Cargo | CRediT | Lidera |
| --- | --- | --- | --- | --- | --- |
| **A1** — Dados, protocolo e integração | Fausto Cananga | 2025102675 | Engenheiro de Dados | Data curation · Software (integração) · Project administration | D1, D3 (interface), D5 |
| **A2** — Classificadores e teto | Gabriela Cardoso | 79295 | Responsável da Baseline | Software · Investigation | D3 (comparação), D4 (Tabela 1) |
| **A3** — Estratégias de aquisição (**porta-voz**) | Gonçalo Vilarinho | 81970 | Investigador | Methodology · Investigation · Visualization | D2, D3 (método avançado) |
| **A4** — Classes raras e enviesamento | Pedro Ribeiro | 79354 | Responsável de Produto | Formal analysis · Software | D3 (aplicação), D4 (Tabela 2) |
| **A5** — Avaliação, calibração e XAI | Marco Maceda | 66694 | Avaliador | Formal analysis · Validation · Visualization | D4 (§5), D5 |

Aplicação (assistente de anotação): **A1** (interface/integração) com **A4** (fila, regra de paragem, produto).
Revisão cruzada de código (PL11): **A2 ↔ A3** e **A4 ↔ A5**; o trabalho do **A1** é revisto pelo grupo na aplicação.
Todos programam, todos escrevem e **todos** têm de saber explicar qualquer parte na defesa (07/01).

## 6. Estrutura

```
DL_P2_TP2-G6/
  README.md            # este ficheiro: problema, dados, licenças, execução, estado
  PROTOCOLO.md         # protocolo experimental (contrato científico do projeto)
  AI_LOG.md            # registo do uso de IA (todos os membros)
  requirements.txt
  configs/             # configurações YAML por experiência: estratégia, seed, L e K
  data/                # FORA DO GIT: recortes; splits.csv e splits.json no git
  src/                 # comum.py, dados.py, extrair.py, probe.py, baseline.py, al.py (DINOv2),
                       # backbones.py + al_ft.py (CNN/ViT com fine-tune), avaliar.py,
                       # redundancia.py, app.py (Gradio: fila de anotação)
  seminario/           # ficha-resumo (1 pág.), plano detalhado e slides pitch
  notebooks/           # exploração — nenhum resultado final nasce aqui
  runs/                # embeddings (.npy, FORA do git) + JSON/CSV de cada corrida (no git)
  paper/               # resultados.csv (276 corridas), tabelas prontas e figuras
  docs/                # P2-D1.pdf (proposta), DATASETS, DRONE, RESULTADOS, RESUMO_EXECUTIVO
  scripts/             # contar_codebrim.py, sanidade_knn.py, registar_corridas.py, gerar_pdf_p2d1.py,
                       # afinar_ft.py (hiperparâmetros do fine-tune), comparar_backbones.py
```

## 7. Como executar

```bash
pip install -r requirements.txt

python src/dados.py --verificar --gerar-json   # contagens, fugas e L0 com hash (13 s)
python src/dados.py --determinismo             # mesmo 1.º lote com a mesma seed
python src/extrair.py --modelo dinov2_vits14   # 7 677 × 384, 475 s em CPU, retomável
python scripts/sanidade_knn.py                 # sanidade do extrator: k-NN 0,7000 (acaso 0,2551)
python src/baseline.py --experiencia ambos     # teto (100%) e L-só (5%)
python src/al.py --seeds 0 1 2 --saida runs/al_porclasse   # DINOv2: 6 estratégias × 3 seeds
python src/avaliar.py --B 400                  # Tabelas 1–3 + figura com IC (um backbone; ver --raiz)
python scripts/registar_corridas.py            # reconstrói as linhas de AL no resultados.csv
python src/redundancia.py                      # τ de quase-duplicados (eixo drone)
python src/app.py --fila 40                    # fila de anotação por incerteza
python src/app.py --servir                     # app Gradio (pip install gradio)
```

Cada corrida demora ~2,5 min por estratégia em CPU (20 núcleos, 3 threads por processo);
`src/al.py` aceita `--estrategias` e `--seeds` para correr uma de cada vez.

### 7.2 Três backbones (Emenda 1 do PROTOCOLO)

CNN e Transformer com *fine-tune* em cada ronda (`src/al_ft.py`); DINOv2 congelado continua em `src/al.py`.
Requer **GPU** (torch com CUDA — numa RTX 50xx: `pip install --force-reinstall --no-deps "torch==2.14.1+cu130" torchvision --index-url https://download.pytorch.org/whl/cu130`).

```bash
python src/al_ft.py --backbone cnn --rapido                # teste de fumo (~3 min em CPU)
python scripts/afinar_ft.py --backbone cnn                 # hiperparâmetros (só validação)
python src/al_ft.py --backbone cnn --teto --seeds 0 1 2    # teto 100% por backbone (~4 min cada)
python src/al_ft.py --backbone vit --teto --seeds 0 1 2
python src/al_ft.py --backbone cnn --seeds 0 1 2           # 18 corridas, ~5,7 h numa RTX 5060
python src/al_ft.py --backbone vit --seeds 0 1 2           # 18 corridas, ~3,9 h
python src/al.py --seeds 0 1 2 --saida runs/al_porclasse   # DINOv2 congelado, L por classe (~21 min)
python scripts/comparar_backbones.py                       # tabelas + figura dos 3 backbones
```

Saídas: `runs/al_ft/<backbone>/<estrategia>/seed<s>/` (mesmo formato que `runs/al/`) e `runs/baseline_ft/<backbone>/`.
A 1.ª execução cria `runs/cache_img224.npy` (≈1,1 GB, fora do git).

## 7.3 Resultados finais — 3 backbones (16/11/2026)

Macro-F1 no teste, média de 3 seeds; IC 95% por bootstrap cruzado (imagens × seeds), B=400.

| Backbone | Teto (100%) | Aleatória a 40% | Melhor AL a 40% | Rótulos p/ 95% do teto (aleatória → melhor AL) | Estratégias que batem a aleatória |
| --- | --- | --- | --- | --- | --- |
| CNN — EfficientNet-B0 (fine-tune) | 0,866 | 0,849 | 0,872 | 18% → 12% | **5/5** (ΔALC +0,018 a +0,021) |
| Transformer — ViT-S/16 (fine-tune) | 0,888 | 0,871 | 0,894 | 14% → 10% | **5/5** (ΔALC +0,010 a +0,017) |
| DINOv2 ViT-S/14 (congelado) | 0,831 | 0,813 | 0,831 | 15% → 11% | **3/5** (BALD, entropia, margem) |

Tabela completa em [`paper/tabelas/comparacao_backbones.md`](paper/tabelas/comparacao_backbones.md),
figura em [`paper/figuras/curvas_3_backbones.png`](paper/figuras/curvas_3_backbones.png), tabelas por backbone em
`paper/tabelas/{cnn,vit,dino}/` e leitura em [`docs/RESULTADOS.md`](docs/RESULTADOS.md).

## 7.1 (Histórico) Resultados das primeiras corridas — 27/10, só DINOv2

Capacidade de referência: cabeça linear sobre DINOv2 ViT-S/14 congelado, 7 677 recortes.

| | Macro-F1 no teste |
| --- | --- |
| k-NN (k=20) na validação — sanidade do extrator | **0,7000** (acaso 0,2551) |
| L-só (5% dos rótulos) | 0,7665 |
| **Teto** (100% dos rótulos) | **0,8306** |

O intervalo entre 5% e 100% é estreito (0,766 → 0,831): com características congeladas
de um ViT grande, 322 recortes já resolvem a maior parte do problema. É a razão pela qual
o ganho do AL tem de ser medido com IC emparelhado e não pelo olho na curva.

Tabela 1 (ALC, rótulos para 95% do teto), Tabela 2 (diferença emparelhada vs aleatória) e
Tabela 3 (calibração) em [`paper/tabelas/`](paper/tabelas/); figura em
[`paper/figuras/curva_al.png`](paper/figuras/curva_al.png); números completos e leitura em
[`docs/RESULTADOS.md`](docs/RESULTADOS.md).

## 8. Licenças de terceiros e uso de IA

- **CODEBRIM:** não comercial; dados **não** versionados neste repositório; citar Mundt et al. (2019).
- **Ultralytics (YOLO):** **AGPL-3.0** — usado apenas no guia de deteção da PL07/Aula 07.
  Se entrar no repositório, **declarar aqui** (alternativa sem AGPL: RetinaNet do KerasHub, Cap. 12.3).
- **DINOv2 (Meta):** pesos sob licença própria da Meta; citar Oquab et al. (2023).
- **Uso de IA:** todas as interações com assistentes vão para [`AI_LOG.md`](AI_LOG.md), incluindo os
  três pedidos obrigatórios do protocolo (semana 8 BALD/BADGE, semana 9 fugas de dados, semana 14
  verificação frase-a-frase da discussão).

## 9. Estado do Projeto

| Data | Marco | Estado |
| --- | --- | --- |
| 22/10 | PL06 — repositório, tabela 2.1, guia de segmentação, app v0 | **concluído** |
| 27/10 | Dados medidos, embeddings DINOv2, teto, 18 corridas de AL, avaliação com IC, app de anotação | **concluído** — ver [`docs/RESULTADOS.md`](docs/RESULTADOS.md) |
| 29/10 | **P2-D1** — proposta oficial (2 páginas no Moodle + pitch de 5 min) | **concluído** — ver [`docs/P2-D1.pdf`](docs/P2-D1.pdf) |
| 05/11 | PL08 — `src/dados.py`, baseline, planeamento do seminário | **em curso** |
| 12–16/11 | **Emenda 1** — CNN, ViT e DINOv2 com as mesmas 6 estratégias, L 5% por classe, 54 corridas na GPU | **concluído** — ver [`docs/RESUMO_ALTERACOES.md`](docs/RESUMO_ALTERACOES.md) |
| 15/11 | Ficha-resumo de 1 página (Moodle) | **atualizada** com os 3 backbones |
| 17/11 | **P2-D2** — Seminário I (Temas A, B, C), 25 min + 5 min | **a ser feito** (em preparação) |
| 03/12 | **P2-D3** — protótipo/aplicação (tag `v1-D3`) | planeado |
| 17/12 | **P2-D4** — avaliação final, Tabelas 1–3 | planeado |
| 04/01 | Relatório + repositório + `AI_LOG.md` | planeado |
| 05/01 · 07/01 | **P2-D5** apresentação · defesa individual | planeado |
