"""Simulador de AL com FINE-TUNING do backbone (CNN e Transformer) — Tema C / AL 1.

Pedido do docente (nov/2026): 3 backbones — CNN, Transformer e DINOv2 (no fim) —
com as mesmas estrategias de AL nos 3. O DINOv2 congelado sobre embeddings ja
existe (`al.py`) e satura depressa (curva plana); aqui o backbone e afinado em
cada ronda, o que da curvas mais ingremes e um teste mais exigente ao AL.

Uma ronda:
  1. treina Rede (backbone pre-treinado + cabeca) do PRE-TREINO com o L atual
     (nao ha warm-start: cada ponto da curva depende so de L, como no congelado);
  2. extrai h(x) penultimo + probabilidades para pool, validacao e teste;
  3. limiares por classe na validacao; metricas no teste (so para a curva);
  4. `al.pontuar` / `al.escolher` sobre (h, cabeca) -> K novos recortes.

As estrategias sao literalmente as mesmas funcoes do pipeline congelado.

Uso:
  python src/al_ft.py --backbone cnn --estrategias aleatoria entropia --seeds 0
  python src/al_ft.py --backbone vit --teto --seeds 0 1 2      # 100% do pool
  python src/al_ft.py --backbone cnn --rapido                  # teste de fumo (CPU)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402
import probe  # noqa: E402
from al import ESTRATEGIAS, escolher, pontuar  # noqa: E402
from backbones import BACKBONES, HIPER_FT, Rede  # noqa: E402
from baseline import ece_macro  # noqa: E402
from dados import escolher_L, excluir_sem_rotulo  # noqa: E402

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Imagens em memoria (uint8, ja redimensionadas): ler 7,7k PNG por ronda seria o gargalo
# --------------------------------------------------------------------------------------
def carregar_imagens(tamanho: int = 224) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    from PIL import Image
    from torchvision import transforms

    df = C.carregar("data/splits.csv")
    df, _ = excluir_sem_rotulo(df)
    df = df.reset_index(drop=True)
    Y = C.matriz_rotulos(df)
    cache = C.RUNS / f"cache_img{tamanho}.npy"
    marca = C.RUNS / f"cache_img{tamanho}.hash"
    h = C.hash_indices(range(len(df))) + "_" + str(pd.util.hash_pandas_object(df.path).sum())
    if cache.exists() and marca.exists() and marca.read_text().strip() == h:
        return np.load(cache, mmap_mode="r"), Y, df
    tf = transforms.Compose([transforms.Resize(tamanho), transforms.CenterCrop(tamanho)])
    C.RUNS.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(len(df), 3, tamanho, tamanho))
    t0 = time.time()
    for i, rel in enumerate(df.path):
        img = tf(Image.open(C.caminho_imagem(rel)).convert("RGB"))
        X[i] = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
        if i % 1000 == 0:
            print(f"  cache imagens {i}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    X.flush()
    marca.write_text(h)
    return np.load(cache, mmap_mode="r"), Y, df


def _aumentar(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Flips horizontal/vertical por amostra (defeitos de betao nao tem orientacao)."""
    n = x.shape[0]
    fh = torch.rand(n, generator=g) < 0.5
    fv = torch.rand(n, generator=g) < 0.5
    x = torch.where(fh.view(-1, 1, 1, 1).to(x.device), x.flip(3), x)
    x = torch.where(fv.view(-1, 1, 1, 1).to(x.device), x.flip(2), x)
    return x


# --------------------------------------------------------------------------------------
# Treino e inferencia
# --------------------------------------------------------------------------------------
def treinar(nome: str, X: np.ndarray, Y: np.ndarray, idx: np.ndarray, seed: int,
            dropout: float = 0.0, epocas: int | None = None, lr_backbone: float | None = None) -> Rede:
    hp = dict(HIPER_FT)
    if epocas is not None:
        hp["epocas"] = epocas
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rede = Rede(nome, Y.shape[1], dropout).to(DISPOSITIVO)
    if lr_backbone is not None:
        rede.lr_backbone = lr_backbone
    opt = torch.optim.AdamW(rede.grupos_otimizador(hp["lr_cabeca"], hp["wd"]))
    passos = hp["epocas"] * int(np.ceil(len(idx) / hp["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[gr["lr"] for gr in opt.param_groups], total_steps=max(1, passos), pct_start=0.1
    )
    perda = nn.BCEWithLogitsLoss()
    amp = DISPOSITIVO.type == "cuda"
    escala = torch.amp.GradScaler("cuda", enabled=amp)
    Yt = torch.from_numpy(Y)
    rede.train()
    for _ in range(hp["epocas"]):
        ordem = idx[torch.randperm(len(idx), generator=g).numpy()]
        for i in range(0, len(ordem), hp["batch"]):
            b = np.sort(ordem[i : i + hp["batch"]])
            if len(b) < 2:  # BatchNorm nao aceita lote de 1
                continue
            xb = _aumentar(torch.from_numpy(X[b]).to(DISPOSITIVO, non_blocking=True), g)
            yb = Yt[b].to(DISPOSITIVO)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DISPOSITIVO.type, dtype=torch.float16, enabled=amp):
                l = perda(rede(xb), yb)
            escala.scale(l).backward()
            escala.step(opt)
            escala.update()
            sched.step()
    rede.eval()
    return rede


@torch.no_grad()
def caracteristicas(rede: Rede, X: np.ndarray, idx: np.ndarray, batch: int = 128) -> np.ndarray:
    rede.eval()
    amp = DISPOSITIVO.type == "cuda"
    saida = []
    for i in range(0, len(idx), batch):
        b = idx[i : i + batch]
        with torch.autocast(DISPOSITIVO.type, dtype=torch.float16, enabled=amp):
            h = rede.caracteristicas(torch.from_numpy(X[b]).to(DISPOSITIVO))
        saida.append(h.float().cpu())
    return torch.cat(saida).numpy()


def _cabeca_cpu(rede: Rede) -> probe.Cabeca:
    c = probe.Cabeca(rede.dim, rede.cabeca.fc.out_features, rede.cabeca.drop.p)
    c.load_state_dict({k: v.detach().float().cpu() for k, v in rede.cabeca.state_dict().items()})
    return c.eval()


def _indices(df: pd.DataFrame, rapido: bool, seed: int):
    pool = np.flatnonzero(df.split.values == "train")
    va = np.flatnonzero(df.split.values == "val")
    te = np.flatnonzero(df.split.values == "test")
    if rapido:
        r = np.random.default_rng(seed)
        pool, va, te = (np.sort(r.choice(a, n, replace=False)) for a, n in ((pool, 400), (va, 120), (te, 120)))
    return pool, va, te


# --------------------------------------------------------------------------------------
# Uma corrida = backbone x estrategia x seed
# --------------------------------------------------------------------------------------
def corrida(nome, estrategia, seed, X, Y, df, fracao_L=0.05, fracao_K=0.025, rondas=14,
            estratificacao="por_classe", rapido=False, raiz=None) -> pd.DataFrame:
    pool, va, te = _indices(df, rapido, seed)
    lab = pool[escolher_L(Y[pool], fracao_L, seed, estratificacao)]
    U = np.setdiff1d(pool, lab)
    K = int(round(fracao_K * len(pool)))
    epocas = 1 if rapido else None
    assert not (set(lab.tolist()) & set(te.tolist())), "V1: teste dentro de L"

    rng = np.random.default_rng(seed)
    destino = (raiz or C.RUNS / "al_ft") / nome / estrategia / f"seed{seed}"
    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / "teste_idx.npy", te)
    np.save(destino / "teste_y.npy", Y[te].astype(np.float32))
    todos = np.concatenate([pool, va, te])
    linhas = []
    for r in range(rondas + 1):
        t0 = time.time()
        rede = treinar(nome, X, Y, np.sort(lab), seed * 100 + r,
                       dropout=(0.2 if estrategia == "bald" else 0.0), epocas=epocas)
        F = np.zeros((len(Y), rede.dim), dtype=np.float32)
        F[todos] = caracteristicas(rede, X, todos)
        cab = _cabeca_cpu(rede)
        del rede
        if DISPOSITIVO.type == "cuda":
            torch.cuda.empty_cache()

        Pv = probe.probabilidades(cab, F[va])
        lim = probe.limiares_por_classe(Y[va], Pv)
        Pt = probe.probabilidades(cab, F[te])
        np.save(destino / f"P_r{r:02d}.npy", Pt.astype(np.float32))
        np.save(destino / f"lim_r{r:02d}.npy", lim)
        pc = C.f1_por_classe(Y[te], Pt, lim)
        linhas.append({
            "backbone": nome, "estrategia": estrategia, "seed": seed, "ronda": r,
            "n_rotulos": int(len(lab)), "fracao": round(len(lab) / len(pool), 4),
            "f1_val": round(C.macro_f1(Y[va], Pv, lim), 4),
            "f1_teste": round(C.macro_f1(Y[te], Pt, lim), 4),
            "ece": round(ece_macro(Y[te], Pt), 4),
            "n_candidatos": int(len(U)), "hash_rotulado": C.hash_indices(lab),
            "segundos": round(time.time() - t0, 1),
            **{f"f1_{c}": round(pc[c], 4) for c in C.CLASSES},
        })
        print(f"    {nome}/{estrategia} s{seed} r{r:02d} n={len(lab)} F1={linhas[-1]['f1_teste']:.3f} "
              f"({linhas[-1]['segundos']:.0f}s)", flush=True)
        if r == rondas or len(U) == 0:
            break
        score = pontuar(estrategia, cab, F, U, rng, seed)
        novos = np.setdiff1d(np.asarray(escolher(estrategia, score, U, K, F, np.sort(lab), rng)), lab)
        lab = np.union1d(lab, novos)
        U = np.setdiff1d(pool, lab)
        assert not (set(lab.tolist()) & set(te.tolist())), "V1: o teste entrou no conjunto rotulado"

    res = pd.DataFrame(linhas)
    res.to_csv(destino / "rondas.csv", index=False)
    C.escrever_json({"backbone": nome, "modelo_timm": BACKBONES[nome][0], "estrategia": estrategia,
                     "seed": seed, "rondas": rondas, "K": K, "n_pool": int(len(pool)),
                     "estratificacao": estratificacao, "hiper": HIPER_FT, "rapido": rapido,
                     "final": res.iloc[-1].to_dict()}, destino / "meta.json")
    if not rapido:
        C.registar_resultado({
            "data": time.strftime("%Y-%m-%d %H:%M"), "commit": C.commit_atual(),
            "configuracao": f"al_ft/{nome}/{estrategia}", "estrategia": estrategia, "seed": seed,
            "fracao_rotulos": round(float(res.fracao.iloc[-1]), 4),
            "f1_val": float(res.f1_val.iloc[-1]), "f1_teste": float(res.f1_teste.iloc[-1]),
            "n_rotulos": int(res.n_rotulos.iloc[-1]), "tempo_s": round(float(res.segundos.sum()), 1),
        }, substituir=True)
    return res


def teto(nome, seed, X, Y, df, rapido=False) -> dict:
    """Referencia a 100% do pool para este backbone (normaliza o ALC entre backbones)."""
    pool, va, te = _indices(df, rapido, seed)
    t0 = time.time()
    rede = treinar(nome, X, Y, pool, seed, epocas=1 if rapido else None)
    F = np.zeros((len(Y), rede.dim), dtype=np.float32)
    F[np.concatenate([va, te])] = caracteristicas(rede, X, np.concatenate([va, te]))
    cab = _cabeca_cpu(rede)
    Pv, Pt = probe.probabilidades(cab, F[va]), probe.probabilidades(cab, F[te])
    lim = probe.limiares_por_classe(Y[va], Pv)
    out = {"backbone": nome, "seed": seed, "n": int(len(pool)),
           "f1_val": C.macro_f1(Y[va], Pv, lim), "f1_teste": C.macro_f1(Y[te], Pt, lim),
           "ece": ece_macro(Y[te], Pt), "segundos": round(time.time() - t0, 1)}
    destino = C.RUNS / ("baseline_ft_rapido" if rapido else "baseline_ft") / nome
    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / f"teto_P_s{seed}.npy", Pt.astype(np.float32))
    C.escrever_json(out, destino / f"teto_s{seed}.json")
    print(f"  teto {nome} s{seed}: F1 teste {out['f1_teste']:.3f}  ({out['segundos']:.0f}s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--estrategias", nargs="+", default=ESTRATEGIAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rondas", type=int, default=14)
    ap.add_argument("--L", type=float, default=0.05)
    ap.add_argument("--K", type=float, default=0.025)
    ap.add_argument("--estratificacao", choices=["por_classe", "raro"], default="por_classe")
    ap.add_argument("--teto", action="store_true", help="so treina a 100%% do pool")
    ap.add_argument("--rapido", action="store_true", help="teste de fumo: subconjunto, 1 epoca, 2 rondas")
    a = ap.parse_args(argv)
    if a.rapido:
        a.rondas = min(a.rondas, 2)

    print(f"dispositivo: {DISPOSITIVO}" + ("" if DISPOSITIVO.type == "cuda" else
          "  (AVISO: sem GPU o fine-tune e lentissimo; instalar torch com CUDA)"))
    X, Y, df = carregar_imagens(HIPER_FT["tamanho"])
    t0 = time.time()
    for s in a.seeds:
        if a.teto:
            teto(a.backbone, s, X, Y, df, a.rapido)
            continue
        for e in a.estrategias:
            corrida(a.backbone, e, s, X, Y, df, a.L, a.K, a.rondas, a.estratificacao, a.rapido,
                    raiz=(C.RUNS / "al_ft_rapido") if a.rapido else None)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
