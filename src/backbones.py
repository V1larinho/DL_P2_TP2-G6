"""Backbones treinaveis para o AL com fine-tuning (CNN e Transformer).

O terceiro backbone, DINOv2 ViT-S/14 congelado, continua no pipeline original
(`extrair.py` + `al.py` sobre embeddings): e a condicao de referencia, "no fim".

Cada rede = extrator timm (sem cabeca) + `probe.Cabeca` (Linear com dropout opcional).
Usar a MESMA classe de cabeca do pipeline congelado permite reutilizar sem alteracoes
`al.pontuar`/`al.escolher`: depois de cada ronda extraem-se as caracteristicas
penultimas h(x) do modelo afinado e as estrategias veem exatamente (h, cabeca),
como no caso congelado. Assim as 6 estrategias sao identicas nos 3 backbones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import Cabeca  # noqa: E402

# nome curto -> (modelo timm, lr do backbone). lr da cabeca e comum (HIPER_FT).
BACKBONES = {
    "cnn": ("efficientnet_b0.ra_in1k", 1e-3),                      # CNN, ImageNet-1k
    "vit": ("vit_small_patch16_224.augreg_in21k_ft_in1k", 1e-4),   # Transformer supervisionado
}

# Hiperparametros do fine-tune, CONGELADOS na validacao com a estrategia aleatoria
# (PROTOCOLO.md §2 e Emenda 1; grelha em scripts/afinar_ft.py, resultados em
# runs/afinacao_ft/*.csv). Re-treina-se do pre-treino em cada ronda.
HIPER_FT = dict(epocas=15, batch=32, lr_cabeca=1e-3, wd=1e-4, tamanho=224)


class Rede(nn.Module):
    def __init__(self, nome: str, n_classes: int, dropout: float = 0.0, pretrained: bool = True):
        super().__init__()
        import timm

        modelo, self.lr_backbone = BACKBONES[nome]
        self.extrator = timm.create_model(modelo, pretrained=pretrained, num_classes=0)
        self.dim = self.extrator.num_features
        self.cabeca = Cabeca(self.dim, n_classes, dropout)
        cfg = timm.data.resolve_data_config({}, model=self.extrator)
        self.register_buffer("media", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("desvio", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)

    def normalizar(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float() / 255.0 - self.media) / self.desvio

    def caracteristicas(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.extrator(self.normalizar(x_uint8))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.cabeca(self.caracteristicas(x_uint8))

    def grupos_otimizador(self, lr_cabeca: float, wd: float):
        return [
            {"params": self.extrator.parameters(), "lr": self.lr_backbone, "weight_decay": wd},
            {"params": self.cabeca.parameters(), "lr": lr_cabeca, "weight_decay": wd},
        ]
