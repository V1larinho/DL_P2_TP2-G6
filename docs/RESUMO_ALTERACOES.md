# Resumo das alterações — Emenda 1 (nov/2026)

## 1. O que o professor pediu

| # | Pedido | Feito? | Onde |
|---|---|---|---|
| 1 | **3 backbones**: CNN, Transformer e DINOv2 "normal" | ✅ código pronto | `src/backbones.py`, `src/al_ft.py`, `src/al.py` |
| 2 | **As mesmas estratégias de AL** nos 3 backbones | ✅ | as 6 funções de `al.py` são reutilizadas tal e qual |
| 3 | **DINOv2 no fim** (é o mais forte / referência) | ✅ no desenho | DINOv2 congelado = pipeline original, apresentado em último |
| 4 | Curvas menos planas ("estava muito eficiente") | ✅ | teste: CNN/ViT sobem ~0,75 → 0,87–0,89; DINOv2 0,75 → 0,83 |
| 5 | **L inicial 5% por classe** | ✅ | `dados.selecionar_L_por_classe` (usado nos 3 backbones) |
| 6 | Resultados finais das 54 corridas | ✅ | §4 e `docs/RESULTADOS.md` |

## 2. Como estava antes

- Um único backbone: **DINOv2 ViT-S/14 congelado** + cabeça linear sobre embeddings.
- Com só 5% dos rótulos já dava **F1 ≈ 0,75** → curva quase plana → o AL tinha pouco espaço para mostrar ganho.
- L inicial: 5% do pool estratificado pela "classe positiva mais rara" de cada recorte.

## 3. O que foi feito

### Código

| Ficheiro | Estado | O que faz |
|---|---|---|
| `src/dados.py` | alterado | `selecionar_L_por_classe` + `escolher_L(modo)`. Cada classe recebe ≥ 5% dos seus positivos; \|L\| ≈ 312–319 (≈ 5% do pool) |
| `src/backbones.py` | **novo** | `Rede` = extrator timm + mesma `Cabeca` do pipeline congelado. **CNN** = EfficientNet-B0 (ImageNet-1k). **Transformer** = ViT-S/16 (ImageNet-21k→1k) |
| `src/al_ft.py` | **novo** | AL com *fine-tune*: em cada ronda re-treina do pré-treino com o L atual, extrai h(x) e aplica `al.pontuar`/`al.escolher`. `--teto` (100% do pool por backbone), `--rapido` (teste de fumo) |
| `src/al.py` | alterado | DINOv2 congelado passa a usar L por classe (`--estratificacao raro` repõe o antigo) |
| `scripts/afinar_ft.py` | **novo** | grelha de hiperparâmetros só na validação, com a estratégia aleatória |
| `PROTOCOLO.md` | alterado | §8 Emenda 1: motivo, antes/depois, hiperparâmetros congelados |
| `README.md` | alterado | §7.2 com os comandos |

### Desenho experimental

```
3 backbones  ×  6 estratégias  ×  3 seeds  =  54 corridas  (15 pontos cada: 5% → 40%)
 CNN (fine-tune)   aleatória, entropia, margem,
 ViT (fine-tune)   BALD, CoreSet, BADGE
 DINOv2 (congelado, no fim)
```

- Comparação **entre backbones**: ALC normalizado pelo teto **do próprio backbone** + ganho de cada estratégia sobre a aleatória dentro do backbone.

### Hiperparâmetros (afinados na validação, congelados)

| | CNN | ViT |
|---|---|---|
| lr backbone | 1e-3 | 1e-4 |
| épocas por ronda | 15 | 15 |
| comum | AdamW + OneCycle, lr cabeça 1e-3, wd 1e-4, batch 32, 224 px, flips H/V | |
| F1 val L=5% → 20% | 0,751 → 0,841 | 0,765 → 0,859 |

Grelha completa em `runs/afinacao_ft/cnn.csv` e `vit.csv`. Nota honesta: o melhor lr ficou no limite superior da grelha (margem ≤ 0,01).

### Ambiente

- Instalado **torch 2.14.1+cu130** → a RTX 5060 é usada (antes era só CPU).
- Testes de fumo na CPU e na GPU: todas as estratégias correm sem erro nos dois backbones.

## 4. Resultados (16/11/2026)

| Backbone | Teto | Aleatória a 40% | Melhor AL a 40% | Estratégias que batem a aleatória (IC 95% exclui 0) |
|---|---|---|---|---|
| CNN | 0,866 | 0,849 | 0,872 | 5/5 |
| ViT | 0,888 | 0,871 | 0,894 | 5/5 |
| DINOv2 (no fim) | 0,831 | 0,813 | 0,831 | 3/5 (BALD, entropia, margem) |

Gerado por `python scripts/comparar_backbones.py` → `paper/tabelas/comparacao_backbones.md`,
`paper/figuras/curvas_3_backbones.png` e tabelas por backbone em `paper/tabelas/{cnn,vit,dino}/`.
Ficha-resumo e plano do seminário atualizados.

## 5. O que ainda falta

- Relatório final (P2-D5): ainda não há texto do paper no repositório; tabelas e figura já estão prontas.
- A app Gradio continua a usar só o DINOv2 congelado (é o único que re-treina em segundos).
- Opcional: mais seeds (só há 3) e grelha de lr mais larga (o melhor ficou no limite).
