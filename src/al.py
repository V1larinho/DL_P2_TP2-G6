"""Simulador de aprendizagem ativa (Tema C / AL 1) — dono: A3 (Goncalo).

Uma ronda = treinar a cabeca com o conjunto rotulado atual, pontuar o pool nao
rotulado, adquirir `K` recortes, repetir. Nada aqui olha para o teste para decidir
seja o que for: o teste e avaliado por ronda apenas para a curva final
(PROTOCOLO.md §4), e o protocolo tem de estar fechado antes.

Estrategias (PROTOCOLO.md §3):
  aleatoria  — controlo
  entropia   — media das entropias binarias por classe
  margem     — media de |p-0,5| (menor = mais incerto)
  bald       — H(media MC) - media H, com dropout na cabeca (T passagens)
  coreset    — k-center greedy nos embeddings normalizados
  badge      — embedding de gradiente da ultima camada + k-means++ (Ash et al. 2020)

Adaptacao declarada do BADGE a multi-rotulo: g_i = (p_i - yhat_i) (x) h_i, com
yhat = 1[p >= 0,5] (pseudo-rotulo). No caso multi-classe o original usa o argmax;
aqui ha 6 decisoes independentes, por isso usa-se o vetor de pseudo-rotulos.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402
import probe  # noqa: E402
from baseline import HIPER, carregar_caracteristicas, ece_macro  # noqa: E402
from dados import escolher_L  # noqa: E402

ESTRATEGIAS = ["aleatoria", "entropia", "margem", "bald", "coreset", "badge"]


# --------------------------------------------------------------------------------------
# Pontuacao
# --------------------------------------------------------------------------------------
def _kmeans_pp(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Sementes de k-means++ (D^2). Devolve os indices escolhidos."""
    n = len(X)
    centros = [int(rng.integers(n))]
    d2 = ((X - X[centros[0]]) ** 2).sum(1)
    for _ in range(1, min(K, n)):
        prob = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        i = int(rng.choice(n, p=prob))
        centros.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.array(centros)


def pontuar(estrategia: str, modelo, F: np.ndarray, cand: np.ndarray, rng, seed: int) -> np.ndarray:
    """Score por candidato; MAIOR = mais informativo (coreset trata-se a parte)."""
    Fc = F[cand]
    if estrategia == "aleatoria":
        return rng.random(len(cand))
    if estrategia == "entropia":
        return probe.incerteza_entropia(probe.probabilidades(modelo, Fc))
    if estrategia == "margem":
        return -probe.incerteza_margem(probe.probabilidades(modelo, Fc))
    if estrategia == "bald":
        return probe.incerteza_bald(probe.probabilidades_mc(modelo, Fc, T=10))
    if estrategia == "coreset":
        # nao ha pontuacao por amostra: a escolha e greedy no seletor (k-center)
        return np.zeros(len(cand))
    if estrategia == "badge":
        P = probe.probabilidades(modelo, Fc)
        Yhat = (P >= 0.5).astype(np.float32)
        G = ((P - Yhat)[:, :, None] * Fc[:, None, :]).reshape(len(cand), -1)
        return G
    raise ValueError(estrategia)


def escolher(estrategia: str, score, cand: np.ndarray, K: int, F: np.ndarray, lab: np.ndarray, rng) -> np.ndarray:
    """Devolve os indices (globais) a adquirir nesta ronda."""
    if K >= len(cand):
        return cand.copy()
    if estrategia == "coreset":
        # k-center greedy: comeca do conjunto rotulado. Em blocos, senao a matriz
        # (n_cand x n_lab x dim) passa dos GB e o processo morre por memoria.
        dmin = np.full(len(cand), np.inf)
        Fcand = F[cand]
        for i in range(0, len(lab), 256):
            bloco = F[lab[i : i + 256]]
            for j in range(0, len(Fcand), 512):
                d = ((Fcand[j : j + 512, None, :] - bloco[None, :, :]) ** 2).sum(-1)
                dmin[j : j + 512] = np.minimum(dmin[j : j + 512], d.min(1))
        escolhidos = []
        for _ in range(K):
            j = int(np.argmax(dmin))
            escolhidos.append(j)
            dmin = np.minimum(dmin, ((F[cand] - F[cand][j]) ** 2).sum(1))
        return cand[np.array(escolhidos)]
    if estrategia == "badge":
        idx_local = _kmeans_pp(score, K, rng)
        return cand[idx_local]
    return cand[np.argsort(-score)[:K]]


# --------------------------------------------------------------------------------------
# Uma corrida = estrategia x seed
# --------------------------------------------------------------------------------------
def corrida(
    estrategia: str,
    seed: int,
    F: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    fracao_L: float = 0.05,
    fracao_K: float = 0.025,
    rondas: int = 14,
    dedup_grupo: bool = False,
    raiz_saida: Path | None = None,
    estratificacao: str = "por_classe",
) -> pd.DataFrame:
    pool = np.flatnonzero(df.split.values == "train")
    va = np.flatnonzero(df.split.values == "val")
    te = np.flatnonzero(df.split.values == "test")
    grupos = df.grupo.values

    lab = pool[escolher_L(Y[pool], fracao_L, seed, estratificacao)]
    U = np.setdiff1d(pool, lab)
    K = int(round(fracao_K * len(pool)))
    assert not (set(lab.tolist()) & set(te.tolist())), "V1: teste dentro de L"

    rng = np.random.default_rng(seed)
    destino = (raiz_saida or (C.RUNS / "al")) / estrategia / f"seed{seed}"
    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / "teste_idx.npy", te)
    np.save(destino / "teste_y.npy", Y[te].astype(np.float32))
    linhas = []
    for r in range(rondas + 1):
        t0 = time.time()
        modelo = probe.treinar(F, Y, np.sort(lab), seed=seed * 100 + r, **HIPER, dropout=(0.2 if estrategia == "bald" else 0.0))
        Pv = probe.probabilidades(modelo, F[va])
        lim = probe.limiares_por_classe(Y[va], Pv)
        Pt = probe.probabilidades(modelo, F[te])
        # previsoes de teste por ronda: o bootstrap por imagem (A5) recalcula tudo
        # a partir daqui. Guardar so a metrica impediria qualquer IC honesto.
        np.save(destino / f"P_r{r:02d}.npy", Pt.astype(np.float32))
        np.save(destino / f"lim_r{r:02d}.npy", lim)
        f1v = C.macro_f1(Y[va], Pv, lim)
        f1t = C.macro_f1(Y[te], Pt, lim)
        pc = C.f1_por_classe(Y[te], Pt, lim)

        cand = U
        if dedup_grupo and len(cand):
            ja = set(grupos[lab].tolist())
            cand = np.array([i for i in U if grupos[i] not in ja], dtype=int)
            if len(cand) == 0:
                cand = U

        linhas.append(
            {
                "estrategia": estrategia,
                "seed": seed,
                "ronda": r,
                "n_rotulos": int(len(lab)),
                "fracao": round(len(lab) / len(pool), 4),
                "f1_val": round(f1v, 4),
                "f1_teste": round(f1t, 4),
                "ece": round(ece_macro(Y[te], Pt), 4),
                "n_candidatos": int(len(cand)),
                "hash_rotulado": C.hash_indices(lab),
                "segundos": round(time.time() - t0, 1),
                **{f"f1_{c}": round(pc[c], 4) for c in C.CLASSES},
            }
        )
        if r == rondas:
            break

        score = pontuar(estrategia, modelo, F, cand, rng, seed)
        novos = escolher(estrategia, score, cand, K, F, np.sort(lab), rng)
        novos = np.setdiff1d(np.asarray(novos), lab)
        lab = np.union1d(lab, novos)
        U = np.setdiff1d(pool, lab)
        assert not (set(lab.tolist()) & set(te.tolist())), "V1: o teste entrou no conjunto rotulado"

    res = pd.DataFrame(linhas)
    res.to_csv(destino / "rondas.csv", index=False)
    C.escrever_json(
        {"estrategia": estrategia, "seed": seed, "rondas": rondas, "K": K,
         "n_pool": int(len(pool)), "dedup_grupo": bool(dedup_grupo), "estratificacao": estratificacao,
         "final": res.iloc[-1].to_dict()},
        destino / "meta.json",
    )
    # Uma linha por corrida no registo do PL08 (§P8.7): a corrida inteira fica
    # resumida pelo ultimo ponto da curva. `substituir=True` garante que repetir
    # a mesma (estrategia, seed) atualiza a linha em vez de a duplicar.
    C.registar_resultado(
        {
            "data": time.strftime("%Y-%m-%d %H:%M"),
            "commit": C.commit_atual(),
            "configuracao": f"al/{estrategia}",
            "estrategia": estrategia,
            "seed": seed,
            "fracao_rotulos": round(float(res.fracao.iloc[-1]), 4),
            "f1_val": round(float(res.f1_val.iloc[-1]), 4),
            "f1_teste": round(float(res.f1_teste.iloc[-1]), 4),
            "n_rotulos": int(res.n_rotulos.iloc[-1]),
            "tempo_s": round(float(res.segundos.sum()), 1),
        },
        substituir=True,
    )
    print(
        f"  {estrategia:<10} seed={seed}  F1: {res.f1_teste.iloc[0]:.3f} -> {res.f1_teste.iloc[-1]:.3f}  "
        f"(n {res.n_rotulos.iloc[0]} -> {res.n_rotulos.iloc[-1]})  {res.segundos.sum():.0f}s"
    )
    return res


def main(argv=None) -> int:
    # 6 estrategias correm em paralelo (uma por processo); cada uma fica com poucas
    # threads para nao haver 6 processos a disputar os mesmos nucleos.
    import torch

    torch.set_num_threads(int(os.getenv("AL_THREADS", "3")))
    ap = argparse.ArgumentParser()
    ap.add_argument("--estrategias", nargs="+", default=ESTRATEGIAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rondas", type=int, default=14)
    ap.add_argument("--L", type=float, default=0.05)
    ap.add_argument("--K", type=float, default=0.025)
    ap.add_argument("--dedup-grupo", action="store_true", help="nao readquire outra crop da mesma imagem (regra drone)")
    ap.add_argument("--saida", default=None)
    ap.add_argument("--estratificacao", choices=["por_classe", "raro"], default="por_classe",
                    help="L inicial: 5%% dos positivos de cada classe (novo) ou classe mais rara (antigo)")
    a = ap.parse_args(argv)

    F, Y, df = carregar_caracteristicas()
    pool = int((df.split == "train").sum())
    print(f"pool={pool}  val={int((df.split=='val').sum())}  teste={int((df.split=='test').sum())}  dim={F.shape[1]}")
    print(f"desenho: L={a.L:.1%} ({round(a.L*pool)})  K={a.K:.1%} ({round(a.K*pool)})  rondas={a.rondas}  seeds={a.seeds}")
    saida = Path(a.saida) if a.saida else None
    t0 = time.time()
    for s in a.seeds:
        for e in a.estrategias:
            corrida(e, s, F, Y, df, a.L, a.K, a.rondas, a.dedup_grupo, saida, a.estratificacao)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
