"""Avaliacao: curvas, ALC, rotulos-para-95%, bootstrap por imagem e calibracao — dono: A5 (Marco).

Decisoes que este ficheiro fixa (e que o relatorio tem de citar):

  * **Bootstrap por imagem de origem**, nunca por recorte: os recortes de uma imagem
    nao sao independentes (A5; Loy & Korobova 2023, casos/cluster bootstrap).
  * **Cruzado com as seeds** (2 dimensoes: seeds x imagens), porque as seeds e o
    conjunto de teste sao cruzados e nao aninhados. Com 3 seeds o IC das seeds e
    **aproximado** e isso e dito, nao escondido.
  * **ALC** = trapezios sobre a fracao de rotulos real, normalizada pelo intervalo.
    Curvas nao monotonas nao se suavizam: a descida e um resultado, nao um erro.
  * **Rotulos para 95% do teto**: limiar definido UMA vez pelo teto (100% dos rotulos)
    e aplicado dentro de cada replica; se nao for atingido, e `NaN` e conta como
    nao atingido — nunca 0.
  * **ECE** = media das ECE binarias por classe (calibracao marginal), com 10 bins fixos.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402
from baseline import ece_macro  # noqa: E402


# --------------------------------------------------------------------------------------
# Leitura das corridas
# --------------------------------------------------------------------------------------
def carregar(raiz: Path | None = None, estrategias=None, seeds=(0, 1, 2)) -> dict:
    raiz = raiz or (C.RUNS / "al")
    df = C.carregar("data/splits.csv")
    corridas = {}
    for d in sorted(raiz.glob("*/seed*")):
        est, s = d.parent.name, int(d.name.replace("seed", ""))
        if estrategias and est not in estrategias:
            continue
        if s not in seeds:
            continue
        rondas = pd.read_csv(d / "rondas.csv")
        P = [np.load(d / f"P_r{r:02d}.npy") for r in rondas.ronda]
        lim = [np.load(d / f"lim_r{r:02d}.npy") for r in rondas.ronda]
        corridas[(est, s)] = {
            "df": rondas,
            "P": P,
            "lim": lim,
            "y": np.load(d / "teste_y.npy"),
            "idx": np.load(d / "teste_idx.npy"),
        }
    return corridas, df


def grupos_teste(df: pd.DataFrame, corridas: dict) -> np.ndarray:
    qualquer = next(iter(corridas.values()))
    return df.grupo.values[qualquer["idx"]]


# --------------------------------------------------------------------------------------
# Resumos de uma curva
# --------------------------------------------------------------------------------------
def alc(frac: np.ndarray, f1: np.ndarray) -> float:
    """Area sob a curva, trapezios, normalizada pelo intervalo de fracoes."""
    f, y = np.asarray(frac, float), np.asarray(f1, float)
    if len(f) < 2:
        return float("nan")
    return float(np.trapezoid(y, f) / (f[-1] - f[0]))


def rotulos_para(frac: np.ndarray, f1: np.ndarray, alvo: float) -> float:
    """Primeira fracao em que a curva atinge `alvo` (interpolacao linear). NaN se nao atinge."""
    f, y = np.asarray(frac, float), np.asarray(f1, float)
    ok = np.flatnonzero(y >= alvo)
    if len(ok) == 0:
        return float("nan")
    i = int(ok[0])
    if i == 0:
        return float(f[0])
    if y[i] == y[i - 1]:
        return float(f[i])
    t = (alvo - y[i - 1]) / (y[i] - y[i - 1])
    return float(f[i - 1] + t * (f[i] - f[i - 1]))


def teto_por_seed(seeds, pasta: Path | None = None) -> dict:
    """F1 do teto (100% dos rotulos) por seed, se o baseline ja correu."""
    out = {}
    pasta = pasta or (C.RUNS / "baseline")
    for s in seeds:
        p = pasta / f"teto_s{s}.json"
        if p.exists():
            import json

            out[s] = json.loads(p.read_text(encoding="utf-8"))["f1_teste"]
    return out


# --------------------------------------------------------------------------------------
# Bootstrap cruzado (imagens x seeds)
# --------------------------------------------------------------------------------------
def bootstrap_cruzado(
    corridas: dict,
    grupos: np.ndarray,
    estrategias,
    seeds,
    teto: dict,
    B: int = 400,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    unicos = np.unique(grupos)
    linhas_por_grupo = {g: np.flatnonzero(grupos == g) for g in unicos}
    n_rondas = len(next(iter(corridas.values()))["P"])

    saida = {e: {"alc": [], "l95": [], "final": [], "f1_curva": [], "pc20": []} for e in estrategias}
    for _ in range(B):
        gb = rng.choice(unicos, size=len(unicos), replace=True)
        idx = np.concatenate([linhas_por_grupo[g] for g in gb])
        sb = rng.choice(seeds, size=len(seeds), replace=True)
        for e in estrategias:
            curvas = []
            for s in sb:
                c = corridas.get((e, int(s)))
                if c is None:
                    continue
                y = c["y"][idx]
                f1 = np.array([C.macro_f1(y, c["P"][r][idx], c["lim"][r]) for r in range(n_rondas)])
                curvas.append((c["df"].fracao.values, f1, float(s)))
            if not curvas:
                continue
            frac = curvas[0][0]
            f1m = np.nanmean(np.stack([c[1] for c in curvas]), axis=0)
            # F1 por classe na ronda mais proxima dos 20% (para H2, emparelhado)
            i20 = int(np.argmin(np.abs(frac - 0.20)))
            pcs = np.stack(
                [
                    np.array([C.f1_por_classe(corc["y"][idx], corc["P"][i20][idx], corc["lim"][i20])[k]
                              for k in C.CLASSES])
                    for corc in [corridas[(e, int(s))] for s in sb if (e, int(s)) in corridas]
                ]
            )
            saida[e]["pc20"].append(np.nanmean(pcs, axis=0))
            saida[e]["alc"].append(alc(frac, f1m))
            saida[e]["f1_curva"].append(f1m)
            # alvo do teto: usa o teto do primeiro seed reamostrado (definido fora da replica?)
            alvo = 0.95 * np.nanmean([teto[int(c[2])] for c in curvas if int(c[2]) in teto] or [np.nan])
            saida[e]["l95"].append(rotulos_para(frac, f1m, alvo) if alvo == alvo else np.nan)
            saida[e]["final"].append(f1m[-1])
    return saida


def ic(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


# --------------------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--estrategias", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--fracao-classe", type=float, default=0.20)
    ap.add_argument("--raiz", default=None, help="pasta das corridas (omissao: runs/al)")
    ap.add_argument("--teto", default=None, help="pasta com teto_s*.json (omissao: runs/baseline)")
    ap.add_argument("--nome", default="", help="subpasta de saida em tabelas/ e figuras/ (ex.: cnn)")
    ap.add_argument("--titulo", default="DINOv2 ViT-S/14 congelado")
    a = ap.parse_args(argv)

    raiz = Path(a.raiz) if a.raiz else None
    corridas, df = carregar(raiz=raiz, estrategias=a.estrategias, seeds=tuple(a.seeds))
    if not corridas:
        print(f"sem corridas em {raiz or 'runs/al/'}")
        return 1
    grupos = grupos_teste(df, corridas)
    estrategias = sorted({e for e, _ in corridas})
    teto = teto_por_seed(a.seeds, Path(a.teto) if a.teto else None)
    TAB = C.PAPEL / "tabelas" / a.nome if a.nome else C.PAPEL / "tabelas"
    FIG = C.PAPEL / "figuras" / a.nome if a.nome else C.PAPEL / "figuras"
    print(f"corridas: {len(corridas)}  estrategias: {estrategias}  grupos de teste: {len(np.unique(grupos))}")
    if not teto:
        print("AVISO: sem runs/baseline/teto_s*.json — a coluna 'rotulos para 95%' fica NaN")

    # --- curvas medias por estrategia ------------------------------------------------
    curvas = {}
    for e in estrategias:
        seeds_e = [s for (ee, s) in corridas if ee == e]
        M = np.stack([corridas[(e, s)]["df"].f1_teste.values for s in seeds_e])
        frac = corridas[(e, seeds_e[0])]["df"].fracao.values
        curvas[e] = (frac, np.nanmean(M, 0), np.nanstd(M, 0), len(seeds_e))

    B = a.B
    bs = bootstrap_cruzado(corridas, grupos, estrategias, tuple(a.seeds), teto, B=B)
    linhas = []
    for e in estrategias:
        f, m, sd, ns = curvas[e]
        alcm, alclo, alchi = ic(bs[e]["alc"])
        l95m, l95lo, l95hi = ic(bs[e]["l95"])
        atingiu = np.mean(~np.isnan(np.asarray(bs[e]["l95"], float)))
        fm, flo, fhi = ic(bs[e]["final"])
        linhas.append(
            {
                "estrategia": e,
                "seeds": ns,
                "ALC": alcm,
                "ALC_lo": alclo,
                "ALC_hi": alchi,
                "rot95": l95m,
                "rot95_lo": l95lo,
                "rot95_hi": l95hi,
                "rot95_atingido": atingiu,
                "F1_final": fm,
                "F1_final_lo": flo,
                "F1_final_hi": fhi,
                "ALC_bruto": alc(f, m),
                "ALC_norm": alcm / np.mean(list(teto.values())) if teto else np.nan,
            }
        )
    tab = pd.DataFrame(linhas).sort_values("ALC", ascending=False)
    C.garantir_dirs()
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    with open(TAB / "tabela1.md", "w", encoding="utf-8") as fh:
        fh.write("# Tabela 1 — ALC e rotulos necessarios para 95% do teto\n\n")
        fh.write(f"Bootstrap cruzado (imagens x seeds), B={B}. IC 95% percentil.\n\n")
        fh.write("| Estrategia | ALC | IC 95% | Rotulos p/ 95% do teto | IC 95% | Atingido | F1 final | IC 95% |\n")
        fh.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for _, r in tab.iterrows():
            rot = "NaN" if r.rot95 != r.rot95 else f"{r.rot95:.3f}"
            rot_ic = "—" if r.rot95 != r.rot95 else f"[{r.rot95_lo:.3f}; {r.rot95_hi:.3f}]"
            fh.write(
                f"| {r.estrategia} | {r.ALC:.4f} | [{r.ALC_lo:.4f}; {r.ALC_hi:.4f}] | {rot} | {rot_ic} | "
                f"{100*r.rot95_atingido:.0f}% | {r.F1_final:.4f} | [{r.F1_final_lo:.4f}; {r.F1_final_hi:.4f}] |\n"
            )
        if teto:
            fh.write(f"\nTeto (100% dos rotulos): F1 = {np.mean(list(teto.values())):.4f} "
                     f"(seeds {list(teto)})\n")
    print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nescrito {C.PAPEL / 'tabelas' / 'tabela1.md'}")

    # --- diferencas emparelhadas contra a aleatoria (a decisao da H1) ---------------
    if "aleatoria" in bs and len(bs["aleatoria"]["alc"]) > 10:
        base_alc = np.array(bs["aleatoria"]["alc"], float)
        base_pc = np.stack(bs["aleatoria"]["pc20"])
        difs = []
        for e in estrategias:
            if e == "aleatoria":
                continue
            d = np.array(bs[e]["alc"], float) - base_alc
            m, lo, hi = ic(d)
            dpc = np.stack(bs[e]["pc20"]) - base_pc
            linha = {"estrategia": e, "d_ALC": m, "d_lo": lo, "d_hi": hi, "P(>0)": float(np.mean(d > 0))}
            for k, c in enumerate(C.CLASSES):
                dm, dlo, dhi = ic(dpc[:, k])
                linha[f"d_{c}"] = dm
                linha[f"d_{c}_lo"] = dlo
                linha[f"d_{c}_hi"] = dhi
            difs.append(linha)
        dd = pd.DataFrame(difs).sort_values("d_ALC", ascending=False)
        with open(TAB / "tabela2_diferencas.md", "w", encoding="utf-8") as fh:
            fh.write("# Diferenca emparelhada vs aleatoria (mesmas imagens e seeds reamostradas)\n\n")
            fh.write(f"B={B}. IC 95% percentil da diferenca. **Negrito** = IC exclui 0.\n\n")
            fh.write("| Estrategia | d ALC | IC 95% | P(>0) | " + " | ".join(f"d {c}" for c in C.CLASSES) + " |\n")
            fh.write("| --- | --- | --- | --- | " + " | ".join("---" for _ in C.CLASSES) + " |\n")
            for _, r in dd.iterrows():
                alcd = f"{r.d_ALC:+.4f}"
                alcd = f"**{alcd}**" if (r.d_lo > 0 or r.d_hi < 0) else alcd
                cells = []
                for c in C.CLASSES:
                    v, lo, hi = r[f"d_{c}"], r[f"d_{c}_lo"], r[f"d_{c}_hi"]
                    t = f"{v:+.3f}"
                    cells.append(f"**{t}**" if (lo > 0 or hi < 0) else t)
                fh.write(f"| {r.estrategia} | {alcd} | [{r.d_lo:+.4f}; {r.d_hi:+.4f}] | {r['P(>0)']:.2f} | " + " | ".join(cells) + " |\n")
        print("\nDiferenca emparelhada vs aleatoria (ALC):")
        print(dd[["estrategia", "d_ALC", "d_lo", "d_hi", "P(>0)", "d_Efflorescence", "d_Efflorescence_lo", "d_Efflorescence_hi"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        print(f"escrito {C.PAPEL / 'tabelas' / 'tabela2_diferencas.md'}")

    # --- F1 por classe a uma fracao fixa (H2) ---------------------------------------
    alvo_f = a.fracao_classe
    pc = []
    for e in estrategias:
        seeds_e = [s for (ee, s) in corridas if ee == e]
        for s in seeds_e:
            d = corridas[(e, s)]["df"]
            i = int((d.fracao - alvo_f).abs().idxmin())
            linha = {"estrategia": e, "seed": s, "fracao": d.fracao.iloc[i]}
            linha.update({c: d[f"f1_{c}"].iloc[i] for c in C.CLASSES})
            pc.append(linha)
    pcd = pd.DataFrame(pc)
    med = pcd.groupby("estrategia")[C.CLASSES].mean().sort_index()
    med.to_csv(TAB / "tabela2_por_classe.csv")
    print(f"\nF1 por classe a ~{alvo_f:.0%} dos rotulos:\n{med.to_string(float_format=lambda v: f'{v:.3f}')}")

    # --- calibracao (H4 / Tabela 3) -------------------------------------------------
    cal = []
    for e in estrategias:
        seeds_e = [s for (ee, s) in corridas if ee == e]
        E = np.stack([corridas[(e, s)]["df"].ece.values for s in seeds_e])
        cal.append({"estrategia": e, "ece_inicial": np.nanmean(E[:, 0]), "ece_final": np.nanmean(E[:, -1])})
    cald = pd.DataFrame(cal).sort_values("ece_final")
    with open(TAB / "tabela3_calibracao.md", "w", encoding="utf-8") as fh:
        fh.write("# Tabela 3 — calibracao (ECE marginal por classe, 10 bins)\n\n")
        fh.write("| Estrategia | ECE com 5% dos rotulos | ECE com 40% |\n| --- | --- | --- |\n")
        for _, r in cald.iterrows():
            fh.write(f"| {r.estrategia} | {r.ece_inicial:.4f} | {r.ece_final:.4f} |\n")
    print(f"\nECE:\n{cald.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")

    # --- figura ---------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
        for e in estrategias:
            f, m, sd, _ = curvas[e]
            ax.plot(f * 100, m, marker="o", ms=3, lw=1.6, label=e)
            ax.fill_between(f * 100, m - sd, m + sd, alpha=0.13)
        if teto:
            ax.axhline(np.mean(list(teto.values())), ls="--", c="k", lw=1, label="teto (100%)")
        ax.set_xlabel("rotulos usados (% do pool)")
        ax.set_ylabel("Macro-F1 no teste")
        ax.set_title(f"Aprendizagem ativa — CODEBRIM, {a.titulo}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "curva_al.png")
        print(f"figura: {C.PAPEL / 'figuras' / 'curva_al.png'}")
    except Exception as exc:  # figura nao pode derrubar a avaliacao
        print(f"aviso: figura nao gerada ({exc})")

    C.escrever_json(
        {
            "B": B,
            "seeds": list(a.seeds),
            "teto": teto,
            "titulo": a.titulo,
            "curvas": {e: {"fracao": curvas[e][0].tolist(), "f1": curvas[e][1].tolist(), "sd": curvas[e][2].tolist()} for e in estrategias},
            "diferencas": dd.to_dict(orient="records") if "aleatoria" in bs and len(bs["aleatoria"]["alc"]) > 10 else [],
            "tabela1": tab.to_dict(orient="records"),
            "por_classe": med.to_dict(),
            "calibracao": cald.to_dict(orient="records"),
        },
        TAB / "resumo_avaliacao.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
