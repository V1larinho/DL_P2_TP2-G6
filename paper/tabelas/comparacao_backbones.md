# Comparacao entre backbones — ALC normalizado pelo teto do proprio backbone

Bootstrap cruzado (imagens x seeds), B=400. dALC = diferenca emparelhada vs aleatoria; **negrito** = IC 95% exclui 0.

| Backbone | Teto F1 | Estrategia | ALC | ALC/teto | F1 a 40% | dALC vs aleatoria | IC 95% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | aleatoria | 0.8217 | 0.949 | 0.8489 | — | — |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | entropia | 0.8422 | 0.973 | 0.8675 | **+0.0206** | [+0.0110; +0.0316] |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | margem | 0.8428 | 0.973 | 0.8636 | **+0.0211** | [+0.0126; +0.0301] |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | bald | 0.8397 | 0.970 | 0.8673 | **+0.0180** | [+0.0097; +0.0259] |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | coreset | 0.8428 | 0.973 | 0.8717 | **+0.0211** | [+0.0098; +0.0319] |
| CNN (EfficientNet-B0, fine-tune) | 0.866 | badge | 0.8423 | 0.973 | 0.8688 | **+0.0206** | [+0.0115; +0.0298] |
| Transformer (ViT-S/16, fine-tune) | 0.888 | aleatoria | 0.8521 | 0.960 | 0.8707 | — | — |
| Transformer (ViT-S/16, fine-tune) | 0.888 | entropia | 0.8625 | 0.971 | 0.8831 | **+0.0103** | [+0.0016; +0.0198] |
| Transformer (ViT-S/16, fine-tune) | 0.888 | margem | 0.8649 | 0.974 | 0.8875 | **+0.0127** | [+0.0058; +0.0213] |
| Transformer (ViT-S/16, fine-tune) | 0.888 | bald | 0.8687 | 0.979 | 0.8882 | **+0.0166** | [+0.0078; +0.0246] |
| Transformer (ViT-S/16, fine-tune) | 0.888 | coreset | 0.8690 | 0.979 | 0.8936 | **+0.0169** | [+0.0083; +0.0259] |
| Transformer (ViT-S/16, fine-tune) | 0.888 | badge | 0.8643 | 0.974 | 0.8842 | **+0.0122** | [+0.0030; +0.0211] |
| DINOv2 ViT-S/14 (congelado) | 0.831 | aleatoria | 0.7941 | 0.956 | 0.8128 | — | — |
| DINOv2 ViT-S/14 (congelado) | 0.831 | entropia | 0.8095 | 0.975 | 0.8268 | **+0.0154** | [+0.0053; +0.0258] |
| DINOv2 ViT-S/14 (congelado) | 0.831 | margem | 0.8093 | 0.974 | 0.8249 | **+0.0151** | [+0.0007; +0.0275] |
| DINOv2 ViT-S/14 (congelado) | 0.831 | bald | 0.8119 | 0.977 | 0.8314 | **+0.0178** | [+0.0064; +0.0284] |
| DINOv2 ViT-S/14 (congelado) | 0.831 | coreset | 0.7955 | 0.958 | 0.8189 | +0.0014 | [-0.0105; +0.0146] |
| DINOv2 ViT-S/14 (congelado) | 0.831 | badge | 0.8020 | 0.966 | 0.8274 | +0.0079 | [-0.0045; +0.0198] |
