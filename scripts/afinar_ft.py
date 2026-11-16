"""Afinacao dos hiperparametros de fine-tune (Emenda 1 do PROTOCOLO) — so VALIDACAO.

Regra do PROTOCOLO §2: escolher uma vez, com a estrategia ALEATORIA, e congelar.
Avalia-se cada combinacao (lr do backbone x epocas) em dois tamanhos de L
representativos da curva — o L inicial (5% por classe) e um L aleatorio de 20%
do pool — e escolhe-se a de maior macro-F1 medio na validacao. O teste nunca e lido.

Uso:
  python scripts/afinar_ft.py --backbone cnn
  python scripts/afinar_ft.py --backbone vit
Saida: runs/afinacao_ft/<backbone>.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import al_ft as A  # noqa: E402
import comum as C  # noqa: E402
import probe  # noqa: E402
from dados import escolher_L  # noqa: E402

GRELHA = {
    "cnn": {"lr_backbone": [1e-4, 3e-4, 1e-3], "epocas": [8, 15]},
    "vit": {"lr_backbone": [2e-5, 5e-5, 1e-4], "epocas": [8, 15]},
}


def avaliar(nome, X, Y, idx, va, seed, lr, ep) -> float:
    rede = A.treinar(nome, X, Y, np.sort(idx), seed, epocas=ep, lr_backbone=lr)
    F = A.caracteristicas(rede, X, va)
    cab = A._cabeca_cpu(rede)
    Pv = probe.probabilidades(cab, F)
    return C.macro_f1(Y[va], Pv, probe.limiares_por_classe(Y[va], Pv))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(GRELHA), required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    X, Y, df = A.carregar_imagens()
    pool = np.flatnonzero(df.split.values == "train")
    va = np.flatnonzero(df.split.values == "val")
    rng = np.random.default_rng(a.seed)
    tamanhos = {
        "L5_por_classe": pool[escolher_L(Y[pool], 0.05, a.seed, "por_classe")],
        "L20_aleatorio": np.sort(rng.choice(pool, int(round(0.20 * len(pool))), replace=False)),
    }
    g = GRELHA[a.backbone]
    linhas = []
    for lr, ep in itertools.product(g["lr_backbone"], g["epocas"]):
        for nomeL, idx in tamanhos.items():
            t0 = time.time()
            f1 = avaliar(a.backbone, X, Y, idx, va, a.seed, lr, ep)
            linhas.append({"backbone": a.backbone, "lr_backbone": lr, "epocas": ep, "L": nomeL,
                           "n": len(idx), "f1_val": round(f1, 4), "segundos": round(time.time() - t0, 1)})
            print(f"  {a.backbone} lr={lr:.0e} ep={ep:>2} {nomeL:<14} n={len(idx):>4} F1val={f1:.3f} "
                  f"({linhas[-1]['segundos']:.0f}s)", flush=True)
    res = pd.DataFrame(linhas)
    destino = C.RUNS / "afinacao_ft"
    destino.mkdir(parents=True, exist_ok=True)
    res.to_csv(destino / f"{a.backbone}.csv", index=False)
    media = res.groupby(["lr_backbone", "epocas"]).f1_val.mean().sort_values(ascending=False)
    print("\nmedia nos dois tamanhos de L (validacao):")
    print(media.round(4).to_string())
    lr, ep = media.index[0]
    print(f"\nESCOLHA {a.backbone}: lr_backbone={lr:g} epocas={ep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
