"""Comparacao entre backbones (Emenda 1): CNN e ViT com fine-tune, DINOv2 congelado no fim.

Corre `src/avaliar.py` uma vez por backbone (tabelas/figura em paper/tabelas/<b>/ e
paper/figuras/<b>/) e junta tudo numa tabela e numa figura de 3 paineis.
A comparacao entre backbones usa o ALC normalizado pelo teto DO PROPRIO backbone e o
ganho de cada estrategia sobre a aleatoria dentro do backbone (bootstrap emparelhado).

Uso:  python scripts/comparar_backbones.py [--B 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import avaliar  # noqa: E402
import comum as C  # noqa: E402

BACKBONES = [  # ordem pedida: CNN, Transformer, DINOv2 no fim
    ("cnn", "CNN (EfficientNet-B0, fine-tune)", C.RUNS / "al_ft" / "cnn", C.RUNS / "baseline_ft" / "cnn"),
    ("vit", "Transformer (ViT-S/16, fine-tune)", C.RUNS / "al_ft" / "vit", C.RUNS / "baseline_ft" / "vit"),
    ("dino", "DINOv2 ViT-S/14 (congelado)", C.RUNS / "al_porclasse", C.RUNS / "baseline"),
]
ORDEM = ["aleatoria", "entropia", "margem", "bald", "coreset", "badge"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=400)
    a = ap.parse_args(argv)

    res = {}
    for nome, titulo, raiz, teto in BACKBONES:
        print(f"\n######## {titulo} ########", flush=True)
        rc = avaliar.main(["--raiz", str(raiz), "--teto", str(teto), "--nome", nome,
                           "--titulo", titulo, "--B", str(a.B)])
        if rc:
            return rc
        res[nome] = json.loads((C.PAPEL / "tabelas" / nome / "resumo_avaliacao.json").read_text(encoding="utf-8"))

    # ---- tabela conjunta ---------------------------------------------------------
    out = C.PAPEL / "tabelas" / "comparacao_backbones.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("# Comparacao entre backbones — ALC normalizado pelo teto do proprio backbone\n\n")
        fh.write(f"Bootstrap cruzado (imagens x seeds), B={a.B}. dALC = diferenca emparelhada vs aleatoria; "
                 "**negrito** = IC 95% exclui 0.\n\n")
        fh.write("| Backbone | Teto F1 | Estrategia | ALC | ALC/teto | F1 a 40% | dALC vs aleatoria | IC 95% |\n")
        fh.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for nome, titulo, _, _ in BACKBONES:
            r = res[nome]
            teto = np.mean(list(r["teto"].values())) if r["teto"] else float("nan")
            t1 = {x["estrategia"]: x for x in r["tabela1"]}
            dif = {x["estrategia"]: x for x in r["diferencas"]}
            for e in ORDEM:
                if e not in t1:
                    continue
                x = t1[e]
                if e in dif:
                    d = dif[e]
                    ds = f"{d['d_ALC']:+.4f}"
                    ds = f"**{ds}**" if (d["d_lo"] > 0 or d["d_hi"] < 0) else ds
                    di = f"[{d['d_lo']:+.4f}; {d['d_hi']:+.4f}]"
                else:
                    ds, di = "—", "—"
                fh.write(f"| {titulo} | {teto:.3f} | {e} | {x['ALC']:.4f} | {x['ALC_norm']:.3f} | "
                         f"{x['F1_final']:.4f} | {ds} | {di} |\n")
    print(f"\nescrito {out}")

    # ---- figura de 3 paineis -----------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), dpi=160, sharey=True)
    for ax, (nome, titulo, _, _) in zip(axs, BACKBONES):
        r = res[nome]
        for e in ORDEM:
            if e not in r["curvas"]:
                continue
            c = r["curvas"][e]
            f, m, sd = (np.array(c[k]) for k in ("fracao", "f1", "sd"))
            ax.plot(f * 100, m, marker="o", ms=2.5, lw=1.5, label=e,
                    ls="--" if e == "aleatoria" else "-", c="k" if e == "aleatoria" else None)
            ax.fill_between(f * 100, m - sd, m + sd, alpha=0.10)
        if r["teto"]:
            ax.axhline(np.mean(list(r["teto"].values())), ls=":", c="grey", lw=1, label="teto (100%)")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("rotulos usados (% do pool)")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("Macro-F1 no teste")
    axs[-1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    destino = C.PAPEL / "figuras" / "curvas_3_backbones.png"
    fig.savefig(destino)
    print(f"figura: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
