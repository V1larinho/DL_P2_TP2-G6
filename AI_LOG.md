# AI_LOG.md — registo do uso de IA

Obrigatório na UC: **todas** as interações com assistentes de IA que contribuam para o trabalho
entram aqui, com veredicto e verificação. Cada membro mantém as suas linhas; o dono da tarefa
confirma a verificação. Formato das aulas: *Proposta da IA · Funcionou/correta? · O que estava
errado ou em falta · Como verificou*.

**Regra de ouro:** a IA não é fonte. Tudo o que entra no relatório tem de ser confirmado no artigo
original, no dataset ou no output de um script.

---

## 1. Pedidos obrigatórios (Tema C §8 · PL06 §2.5 · PL07 §2.4 · PL08 §2.5)

| # | Semana | Pedido (texto a usar) | Dono | Estado |
| --- | --- | --- | --- | --- |
| **IA-1** | 6 | *"Escreve um README.md para um projeto de Active Learning com o dataset CODEBRIM de defeitos em pontes de betão."* | A1 | **feito — ver §2 linha 05/11** |
| **IA-2** | 7 | *"Escreve um script que converta anotações COCO JSON (ou máscaras) para o formato necessário."* | A1 | **por fazer** — verificar desenhando as caixas convertidas sobre **5 imagens** |
| **IA-3** | 8 | *"Faz-me 5 perguntas difíceis sobre Active Learning pool-based com modelos fundacionais congelados."* | todos | **feito — ver §2 linha 05/11** |
| **IA-4** | 8 | *"Implementa o BADGE (embeddings de gradiente da última camada + k-means++) para um classificador multi-rótulo."* | A3 | **por fazer** — verificar: `K = |pool|` seleciona tudo; escolhidos diversos no t-SNE; mesma seed ⇒ mesmo resultado |
| **IA-5** | 9 | *"Identifica fugas de dados, métricas inadequadas e comparações injustas neste PROTOCOLO.md."* | A1 | **por fazer** — decidir crítica a crítica; resposta já iniciada em [`PROTOCOLO.md`](PROTOCOLO.md) §6 |
| **IA-6** | 14 | *"Marca cada frase desta discussão como suportada ou não pelos números da tabela."* | todos | **por fazer** — reescrever as frases não suportadas |

## 2. Registo (uma linha por interação)

| Data | Membro | Pedido feito | Proposta da IA (código/afirmação) | Funcionou / correta? (S/N/parcial) | O que estava errado ou em falta | Como verificou (output, shape, métrica) | Ficheiro/commit afetado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 27/10 | A3 (Gonçalo) | Aquisição e caracterização quantitativa do CODEBRIM — contagens por classe, verificação de integridade (md5) e deteção de fugas entre partições — com `scripts/contar_codebrim.py` | Script de contagem a partir do zip, com leitura das etiquetas XML e verificação de fuga | **parcial** | Primeira versão contava 7 788 imagens porque incluía os resource forks `__MACOSX/._*.png`; a segunda descobriu **52 recortes com vetor todo a zero**. Também foi preciso tratar `label` como multi-rótulo (`A|B`), porque a pasta só diz `defects` | `md5` = `c1612d9674e2e628e72e7f5817c40130` (igual ao publicado) · `--contar` → 7 677 utilizáveis, 52 excluídos, 0 imagens em duas partições, `exit 0`; `splits.csv` com 7 677 linhas e 29 combinações de rótulos | `scripts/contar_codebrim.py`, `data/splits.csv`, `docs/contagens_codebrim.json` |
| 27/10 | A3 (Gonçalo) | Análise comparativa dos datasets candidatos (CODEBRIM, SegCODEBRIM, dacl10k e alternativas só-de-fissura) face aos requisitos do Tema C: tarefa, classes, licença e dimensão | `docs/DATASETS.md` com tabela C1–C7 e recomendação em 3 níveis | **parcial** | O protocolo do tema nomeia **armadura exposta e corrosão** como classes raras; o paper mostra **eflorescência 833** como a mais rara (armadura exposta 1 507, corrosão 1 559). O protocolo também trata "5 354" como recortes, mas são **caixas** (+ 2 506 de fundo) | Lido o §3.2 e o §4.1 do paper (arXiv 1904.08486) e o registo Zenodo (ficheiros, tamanhos e md5); contagens somam 8 304 > 5 354 caixas, o que é coerente com multi-rótulo | `docs/DATASETS.md`, `README.md`, `docs/P2-D1.md`, `PROTOCOLO.md` |
| 27/10 | A3 (Gonçalo) | Estruturação inicial do repositório e redação preliminar de `PENDENTES.md`, P2-D1, `PROTOCOLO.md` e pitch, com base nos seis enunciados da UC | Esqueleto + documentos listados | **parcial** | Ver [`docs/REVISAO.md`](docs/REVISAO.md): a IA propôs a métrica da primeira proposta (`mAP@0,5:0,95`), que **não se aplica** a um dataset de classificação de recortes; o texto do protocolo confundia "caixas" com "recortes"; a estimativa computacional contava 4 estratégias quando são 5; o pseudo-código do ALC usa `np.trapz` (descontinuado no NumPy ≥ 2.0); "L estratificado por classe" é indefinido em multi-rótulo | Cada achado foi confrontado com a secção de origem do PDF (coluna "Fonte" em `PENDENTES.md`) e com o número de páginas/campos do documento; ver §3 abaixo | `README.md`, `PENDENTES.md`, `docs/P2-D1.md`, `PROTOCOLO.md`, `slides/P2-D1_pitch.md` |
| 27/10 | A1 (Fausto) | Levantamento bibliográfico de datasets de defeitos captados por drone e de limiares publicados para a deteção de quase-duplicados | Recomendação de segundo domínio + τ para deduplicação | **parcial** | Não existe dataset **multi-rótulo** de defeitos captado por drone com licença verificável: os candidatos são segmentação (dacl10k, CC BY-NC; SegCODEBRIM, MIT), binários (SDNET2018, CC BY 4.0) ou de acesso fechado. Para τ **não há valor publicado** para betão; o `imagededup` usa Hamming ≤10 em 64 bits e o 0,95 do CLIP é heurística de issue, não limiar validado | τ medido neste dataset (p99,9 dos pares entre imagens) = **0,868**; 21,5% do pool redundante após L, em `runs/redundancia.json` | `docs/DRONE.md` §5.1, `src/redundancia.py` |
| 27/10 | A2 (Gabriela) | Revisão do protocolo oficial de *linear probe* do DINOv2 e fundamentação metodológica do baseline | SGD momentum 0,9, weight decay 0, cosine, busca de lr 1e-5..0,1; `linear.py` 10 épocas × 1 250 iterações, batch 128/GPU | **S** (para o que foi lido) — mas a adaptação **não** é o protocolo oficial: o `linear.py` é softmax de rótulo único | O `log_regression.py` faz **L2 regularization**, não normalização L2 das características — são coisas diferentes e a confusão é fácil de defender mal. O CODEBRIM original usa **sigmoide + BCE** por classe, que é o que fazemos | `src/probe.py` (BCE, AdamW, lr 0,01, 200 épocas), teto 0,8306 em 3 seeds | `src/probe.py`, `src/baseline.py` |
| 27/10 | A4 (Pedro) | Revisão de padrões de interface para anotação assistida por AL, estimativa realista do custo de anotação e risco de ancoragem do anotador | Mostrar previsão ao anotador para acelerar | **parcial / contra** | Mostrar a previsão **ancora**: em patologia, 7% de decisões corretas foram revertidas por sugestão errada da IA (Rosbach et al.). Não há batch size "ótimo" publicado; e não se pode reportar κ sem dois anotadores independentes | App passa a **ocultar** a previsão por omissão; `--revelar` existe para testar a ancoragem; o custo é tempo observado por recorte, não "segundos por rótulo" | `src/app.py` |
| 27/10 | A5 (Marco) | Definição da metodologia estatística: intervalos de confiança para dados agrupados, ECE multi-rótulo e ALC | Bootstrap simples sobre recortes | **parcial (errado)** | Com vários recortes por imagem, o bootstrap tem de ser **por cluster (imagem de origem)**; seeds e teste são **cruzados**, não aninhados; ECE multi-rótulo faz-se **por classe e média** (top-1 não tem sentido com sigmoides independentes); ALC por trapézios na fração real, **sem suavizar** curvas não monótonas | `src/avaliar.py`: bootstrap cruzado B=400, 135 grupos de teste; Tabelas 1–3 | `src/avaliar.py`, `paper/tabelas/` |
| 27/10 | A3 (Gonçalo) | Recolha de referências primárias para BALD, CoreSet e BADGE aplicados a características congeladas | — | **N (falhou)** | O agente de pesquisa devolveu `upstream_provider_error`; o BADGE foi implementado a partir da fórmula do artigo (Ash et al. 2020) com adaptação multi-rótulo declarada no docstring, **sem** verificação externa. É o único ponto deste lote sem segunda fonte | `pontuar()`/`escolher()` em `src/al.py`; BADGE deu ALC 0,8072 (+0,0128 vs aleatória) | `src/al.py` |
| 27/10 | todos | Execução integral do pipeline experimental: dados → embeddings → teto → 18 corridas de AL → avaliação com IC | Código + números | **S** nos números | Ver [`docs/RESULTADOS.md`](docs/RESULTADOS.md) §7: baseline EfficientNet não corrida, campanha de anotação não feita, protocolo ainda não fechado | `python src/dados.py --verificar` sem erro · k-NN 0,7000 (acaso 0,2551) · teto 0,8306 · BALD +0,0218 [+0,0109; +0,0321] · 270 treinos | `src/`, `paper/tabelas/`, `docs/RESULTADOS.md` |
| 05/11 | A3 (Gonçalo) | IA-1: *"Escreve um README.md para um projeto de Active Learning com o dataset CODEBRIM de defeitos em pontes de betão."* | README com dataset "CODEBRIM Classification Dataset", 7 788 imagens, Macro-F1, estratégias entropia/BALD/CoreSet, resnet50 como extrator | **parcial (errado)** | A IA inventou: 7 788 imagens (são 7 677 utilizáveis; a diferença são 52 recortes sem rótulo + 59 resource forks excluídos); usou resnet50 como extrator em vez de DINOv2; propôs mAP como métrica (não se aplica a classificação de recortes); não distinguiu raridade de criticidade (eflorescência é a mais rara, não a armadura exposta) | Comparado com `docs/contagens_codebrim.json` (7 677 recortes), `docs/DATASETS.md` (extrator DINOv2) e `docs/P2-D1.md` §2 (mAP não aplicável) | `README.md`, `AI_LOG.md` |
| 05/11 | A3 (Gonçalo) | IA-3: *"Faz-me 5 perguntas difíceis sobre Active Learning pool-based com modelos fundacionais congelados."* | 5 perguntas sobre: (1) escolha de K; (2) inicialização de L; (3) calibração; (4) CoreSet vs BADGE; (5) cold start | **parcial** | A IA gerou as perguntas corretamente. Erros nas respostas sugeridas: (1) não há K "ótimo" publicado — depende do dataset; (2) a estratificação em multi-rótulo não está definida no protocolo original (a nossa definição é a "classe positiva mais rara"); (3) a ECE multi-rótulo é marginal por classe, não top-1; (4) CoreSet não usa gradientes — a confusão com BADGE é comum; (5) "cold start" com embeddings fortes não é problemático como com redes treinadas do zero | Respostas verificadas em Ash et al. 2020 (BADGE), Sener & Savarese 2018 (CoreSet), Gal et al. 2017 (BALD), Guo et al. 2017 (ECE), PROTOCOLO.md §3 | `AI_LOG.md`, `docs/RESPOSTAS_PL08.md` |

| 12/11 | A3 (Gonçalo) | Análise crítica das melhorias pedidas pelo docente (`melhorias.md` e notas da aula) e proposta de um desenho experimental que as satisfaça | Comparar três backbones — CNN, Transformer e DINOv2 congelado — com as mesmas seis estratégias de AL, mantendo o DINOv2 como referência final; L inicial de 5% por classe | **S** | — (desenho discutido e aceite pelo grupo antes de implementar) | Confronto ponto a ponto com o pedido do docente; tabela de conformidade em `docs/RESUMO_ALTERACOES.md` §1 | `PROTOCOLO.md` (Emenda 1), `docs/RESUMO_ALTERACOES.md` |
| 12/11 | A3 (Gonçalo) | Implementação da seleção do L inicial com quota mínima de 5% dos positivos de cada classe | `dados.selecionar_L_por_classe` + `escolher_L(modo)` | **S** | — | Cada classe recebe ≥ 5% dos seus positivos; \|L\| = 319 / 315 / 312 nas seeds 0–2; o método antigo (`modo="raro"`) mantém-se reprodutível | `src/dados.py`, `src/al.py` |
| 12/11 | A3 (Gonçalo) | Implementação do AL com *fine-tune* para EfficientNet-B0 e ViT-S/16, reutilizando sem alterações as funções de aquisição do pipeline congelado | `src/backbones.py` (extrator timm + a mesma `Cabeca`) e `src/al_ft.py` (re-treino do pré-treino por ronda, extração de h(x), mesmas `pontuar`/`escolher`) | **parcial** | O modo de teste rápido gravava o teto na pasta dos resultados reais (`runs/baseline_ft`); corrigido para `baseline_ft_rapido` | Teste de fumo em CPU e GPU com as 6 estratégias nos dois backbones; saídas com o mesmo formato de `runs/al/` | `src/backbones.py`, `src/al_ft.py` |
| 12/11 | A3 (Gonçalo) | Configuração do ambiente para treino em GPU (RTX 5060) | `pip install --upgrade torch --index-url …/cu128` | **N** | O comando não teve efeito: a versão CPU já instalada foi considerada atualizada. Foi necessário `--force-reinstall --no-deps "torch==2.14.1+cu130"` | `torch.cuda.is_available()` = True; treino na GPU verificado | `README.md` §7.2 |
| 13/11 | A3 (Gonçalo) | Afinação dos hiperparâmetros do *fine-tune* exclusivamente na validação e com a estratégia aleatória, conforme o PROTOCOLO §2 | Grelha lr × épocas avaliada em L = 5% e L = 20%; escolha: CNN lr 1e-3 / 15 épocas, ViT lr 1e-4 / 15 épocas | **parcial** | O melhor lr ficou no limite superior da grelha nos dois backbones (margem ≤ 0,01 F1); declarado como limitação | `runs/afinacao_ft/cnn.csv` e `vit.csv`; o conjunto de teste nunca foi lido | `scripts/afinar_ft.py`, `src/backbones.py` |
| 15–16/11 | A3 (Gonçalo) | Execução das 54 corridas (3 backbones × 6 estratégias × 3 seeds) e dos tetos a 100% dos rótulos | Encadeamento numa só execução em GPU, com registo em `runs/logs/emenda1.log` | **S** | — | Terminou sem erros (22:35 → 08:51); 54 pastas com `meta.json` e `rondas.csv` | `runs/al_ft/`, `runs/al_porclasse/`, `runs/baseline_ft/` |
| 16/11 | A3 (Gonçalo) | Extensão da avaliação à comparação entre backbones, com ALC normalizado pelo teto de cada backbone e IC por bootstrap | `avaliar.py` com `--raiz/--teto/--nome`; `scripts/comparar_backbones.py` com tabela conjunta e figura de 3 painéis | **parcial** | (1) Um resumo preliminar leu a coluna `f1_val` em vez de `f1_teste` e apresentou números de validação como se fossem de teste — detetado ao comparar com o bootstrap e corrigido; (2) uma substituição automática no código criou uma auto-referência (`TAB = TAB / …`), corrigida antes de correr | Valores finais saídos de `avaliar.py` sobre o teste; consistência verificada entre `rondas.csv`, Tabela 1 e figura | `src/avaliar.py`, `scripts/comparar_backbones.py`, `paper/tabelas/`, `paper/figuras/` |
| 16/11 | A3 (Gonçalo) | Atualização da documentação e do material do seminário com os resultados finais | Novas secções em `README.md`, `docs/RESULTADOS.md`, ficha-resumo e plano do seminário | **S** | Números revistos contra `paper/tabelas/comparacao_backbones.md` | Cada valor citado existe na tabela gerada pelo script | `README.md`, `docs/`, `seminario/` |

*(acrescentar linhas; não apagar — o histórico faz parte do entregável)*

## 3. Fontes externas: o que foi confirmado e o que falta

### Resolvido (27/10, segunda tentativa — pesquisa feita diretamente)

| Facto | Valor confirmado | Fonte lida |
| --- | --- | --- |
| Imagens e pontes | 1 590 imagens de **30 pontes** | arXiv `1904.08486` §3.2 |
| Anotações | **5 354** caixas de defeito + **2 506** caixas de fundo | idem |
| Contagem por classe | fissura 2 507 · delaminação 1 898 · corrosão 1 559 · armadura exposta 1 507 · **eflorescência 833** | idem |
| Divisão oficial | **150** exemplos por classe em validação e em teste; agrupamento **por imagem**, não por ponte | idem §4.1 |
| Ficheiros e versões | `classification_dataset` 7,9 GB (md5 `c1612d96…`) · `original_images` 8,3 GB · `cropped` 7,9 GB · `classification_balanced` 12,2 GB (**não usar**) | [zenodo.org/records/2620293](https://zenodo.org/records/2620293) |
| Extensão | **SegCODEBRIM**: 420 máscaras de fissura, **MIT** | [zenodo.org/records/10071534](https://zenodo.org/records/10071534) |
| Segundo domínio candidato | **dacl10k**: 9 920 imagens, 19 classes (13 danos + 6 objetos) | arXiv `2309.00460` · FiftyOne zoo |

### Resolvido (27/10, pesquisa por membro)

| Facto | Valor confirmado | Fonte lida |
| --- | --- | --- |
| Identificador DINOv2 no timm | `vit_small_patch14_dinov2.lvd142m` (22 M parâmetros; 518 px por omissão → `dynamic_img_size=True` para 224) | executado: `timm 1.0.30`, embedding 384-d |
| Linear probe no DINOv2 oficial | SGD momentum 0,9 · wd 0 · cosine · lr em grelha 1e-5..0,1 · 10 épocas × 1 250 it | [github.com/facebookresearch/dinov2 · eval/linear.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/linear.py) |
| Rotulos do CODEBRIM | **sigmoide por classe + BCE**, limiar 0,5 no paper | arXiv `1904.08486` (procedimento de treino) |
| Datasets de drone | Nenhum multi-rótulo com licença verificável; dacl10k **CC BY-NC 4.0**, SDNET2018 **CC BY 4.0**, SegCODEBRIM **MIT** | [dacl10k-toolkit](https://github.com/phiyodr/dacl10k-toolkit) · [SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) · [SegCODEBRIM](https://zenodo.org/records/10071534) |
| Bootstrap com clusters | Reamostrar **a imagem de origem inteira**, nunca recortes isolados | [Loy & Korobova, *Bootstrapping Clustered Data in R*](https://journal.r-project.org/articles/RJ-2023-015/) |
| ECE multi-rótulo | **Marginal por classe**, depois média; \(p_c=\sigma(z_c/T)\) com T ajustado por BCE na validação | [Guo et al. 2017](https://proceedings.mlr.press/v70/guo17a.html) · [Ullah et al. 2024](https://arxiv.org/html/2411.04276v1) |
| Ancoragem por sugestão da IA | Efeito medido com especialistas (7% de reversão de decisões corretas) — **não** em betão | [Rosbach et al., *Stuck on Suggestions*](https://arxiv.org/html/2603.11821v2) |

### Por confirmar

- Algoritmo e garantia do **CoreSet** e detalhe dos *gradient embeddings* do **BADGE** — o agente de pesquisa falhou
  (`upstream_provider_error`) e a implementação vem do artigo, sem segunda fonte. Re-perguntar antes do seminário.
- Limiar de quase-duplicados **para vídeo de drone** (o 0,868 medido é de imagens fixas do CODEBRIM).

Tudo o resto fica como leitura em `paper/leituras/` (`C1.11` em `PENDENTES.md`), sempre a citar o
artigo original:

- Mundt et al., *Meta-Learning Convolutional Neural Architectures for Multi-Target Concrete Defect
  Classification with the CODEBRIM Dataset*, CVPR 2019.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023 → extrator.
- Sener & Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR 2018.
- Ash et al., *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds*, ICLR 2020.
- Gal et al., *Deep Bayesian Active Learning with Image Data*, ICML 2017.
- Settles, *Active Learning Literature Survey*, 2009.
- Munjal et al., *Towards Robust and Reproducible Active Learning Using Neural Networks*, 2022.
- Jaziri et al., *SegCODEBRIM*, WACV 2024 (extensão opcional).
