"""Pipeline de dados do P2-D2 (PL08 §2.2) — dono: A1 (Fausto).

Requisitos da PL08, cumpridos tal como estao escritos:
  * `carregar_splits(csv, seed)` devolve train_ds, val_ds, test_ds a partir de splits.csv.
  * `verificar(df)` rebenta se um grupo (imagem de origem) estiver em duas particoes.
  * `python src/dados.py --verificar` imprime as contagens por particao e classe.
  * duas execucoes com a mesma seed produzem o mesmo primeiro lote.

Acrescimos que o protocolo exige e a PL08 nao da:
  * `--gerar-json` escreve data/splits.json com os hashes e as listas L0 por seed.
  * `L` estratificado em multi-rotulo (regra fixada aqui, uma so vez, em `selecionar_L`).

Aumentacao so no treino (questao 1 da PL08): augmentar antes da divisao contaminaria
validacao e teste com versoes transformadas das mesmas imagens.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import comum as C  # noqa: E402

# Normalizacao ImageNet: a que os pesos DINOv2 do timm esperam.
MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------
# Leitura e verificacao
# --------------------------------------------------------------------------------------
def verificar(df: pd.DataFrame) -> None:
    """Rebenta (AssertionError) se houver fuga entre particoes. Igual a PL08 §2.2."""
    grupos = {s: set(df[df.split == s].grupo) for s in ["train", "val", "test"]}
    assert not grupos["train"] & grupos["test"], "group in both train and test"
    assert not grupos["val"] & grupos["test"], "group in both validation and test"
    assert not grupos["train"] & grupos["val"], "group in both train and validation"
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


def contar(df: pd.DataFrame, verboso: bool = True) -> dict:
    """Contagens por particao e por classe, a partir da matriz de rotulos."""
    saida = {}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        y = C.matriz_rotulos(d)
        saida[s] = {
            "recortes": int(len(d)),
            "imagens": int(d.grupo.nunique()),
            "por_classe": {c: int(y[:, i].sum()) for i, c in enumerate(C.CLASSES)},
            "multi_rotulo": int((y.sum(1) > 1).sum()),
            "sem_classe_positiva": int((y.sum(1) == 0).sum()),
        }
    if verboso:
        print(f"{'particao':<8}{'recortes':>9}{'imagens':>9}{'multi':>7}{'vazio':>7}")
        for s, v in saida.items():
            print(f"{s:<8}{v['recortes']:>9}{v['imagens']:>9}{v['multi_rotulo']:>7}{v['sem_classe_positiva']:>7}")
        print()
        tab = pd.DataFrame({s: v["por_classe"] for s, v in saida.items()}).T
        tab["TOTAL"] = tab.sum(axis=1)
        print(tab.to_string())
    return saida


def excluir_sem_rotulo(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """52 recortes do zip tem o vetor todo a zero: nao sao exemplos de nada.

    Ficam de fora do pool (nao do ficheiro), porque no AL seriam os primeiros
    escolhidos por "incerteza" — incerteza artificial, sem sinal a aprender.
    """
    y = C.matriz_rotulos(df)
    vazio = y.sum(1) == 0
    return df[~vazio].reset_index(drop=True), int(vazio.sum())


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class Recortes:
    """Dataset minimo: (imagem, vetor de rotulos). `treino=True` liga a aumentacao."""

    def __init__(self, df: pd.DataFrame, treino: bool = False, tamanho: int = 224):
        self.df = df.reset_index(drop=True)
        self.treino = treino
        self.tamanho = tamanho
        self.y = C.matriz_rotulos(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def _transformacoes(self):
        from torchvision import transforms

        if self.treino:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.tamanho, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1, 0.02)], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(MEDIA, DESVIO),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(self.tamanho),
                transforms.CenterCrop(self.tamanho),
                transforms.ToTensor(),
                transforms.Normalize(MEDIA, DESVIO),
            ]
        )

    def __getitem__(self, i: int):
        from PIL import Image

        tf = getattr(self, "_tf", None)
        if tf is None:
            tf = self._tf = self._transformacoes()
        img = Image.open(C.caminho_imagem(self.df.path.iloc[i])).convert("RGB")
        import torch

        return tf(img), torch.from_numpy(self.y[i])


def _semente(seed: int):
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # CPU + shuffle com gerador proprio: basta


def carregar_splits(csv="data/splits.csv", seed=42, batch_size=64, numeros_trabalho=0, tamanho=224):
    """Devolve train_ds, val_ds, test_ds (DataLoaders) a partir de splits.csv."""
    import torch
    from torch.utils.data import DataLoader

    _semente(seed)
    df = C.carregar(csv)
    verificar(df)
    df, _ = excluir_sem_rotulo(df)

    carga = {}
    for s in ("train", "val", "test"):
        ds = Recortes(df[df.split == s], treino=(s == "train"), tamanho=tamanho)
        g = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[s])
        carga[s] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(s == "train"),
            num_workers=numeros_trabalho,
            generator=g if s == "train" else None,
            drop_last=False,
        )
    return carga["train"], carga["val"], carga["test"]


# --------------------------------------------------------------------------------------
# L estratificado em multi-rotulo
# --------------------------------------------------------------------------------------
def estrato(y: np.ndarray) -> np.ndarray:
    """Estrato de cada recorte = classe positiva mais rara que ele tem.

    E a regra que o PROTOCOLO.md §3 manda fixar. Justificacao: garante a presenca
    proporcional das classes escassas sem multiplicar combinacoes (ha 29 combinacoes
    de rotulos, demasiado para estratificar por combinacao com L=322).
    """
    freq = y.sum(0)
    ordem = np.argsort(freq)  # mais rara primeiro
    chave = np.full(len(y), -1, dtype=int)
    for c in reversed(ordem):  # preenche da mais frequente para a mais rara
        tem = y[:, c] > 0
        chave[tem] = c
    return chave


def selecionar_L(y: np.ndarray, fracao: float, seed: int) -> np.ndarray:
    """Escolhe `fracao` dos recortes, por estrato, de forma deterministica.

    Garante pelo menos 1 exemplo por classe presente, e nunca mais do que o estrato tem.
    """
    rng = np.random.default_rng(seed)
    chave = estrato(y)
    n = len(y)
    alvo = int(round(fracao * n))
    escolhidos: list[int] = []
    for k in np.unique(chave):
        idx = np.flatnonzero(chave == k)
        rng.shuffle(idx)
        alvo_k = max(1, int(round(len(idx) / n * alvo)))
        escolhidos.extend(idx[: min(alvo_k, len(idx))].tolist())
    # acerta o total (os arredondamentos por estrato nunca batem exatamente)
    resto = np.setdiff1d(np.arange(n), escolhidos)
    rng.shuffle(resto)
    falta = alvo - len(escolhidos)
    if falta > 0:
        escolhidos.extend(resto[:falta].tolist())
    elif falta < 0:
        rng.shuffle(escolhidos := np.array(escolhidos))
        escolhidos = escolhidos[:alvo].tolist()
    idx = np.array(sorted(int(i) for i in escolhidos))
    assert len(idx) == alvo, (len(idx), alvo)
    return idx


def selecionar_L_por_classe(y: np.ndarray, fracao: float, seed: int) -> np.ndarray:
    """L inicial = `fracao` dos POSITIVOS de cada classe (pedido do docente, nov/2026).

    Para cada classe c: sorteia ceil(fracao * n_pos_c) recortes entre os que tem c,
    sem repor; a uniao das 6 escolhas e o L inicial. Diferente de `selecionar_L`:
    um recorte com 3 defeitos conta para as 3 classes, e cada classe fica com
    pelo menos `fracao` dos seus positivos. Consequencia declarada: |L| e variavel
    (multi-rotulo + sobreposicao), proximo de `fracao * n`.
    """
    import math

    rng = np.random.default_rng(seed)
    escolhidos: set[int] = set()
    for c in range(y.shape[1]):
        pos = np.flatnonzero(y[:, c] > 0)
        if len(pos) == 0:
            continue
        rng.shuffle(pos)
        alvo = max(1, math.ceil(fracao * len(pos)))
        # os que ja entraram por outra classe contam para esta (nao se duplica a quota)
        ja = sum(1 for i in pos if int(i) in escolhidos)
        for i in pos:
            if ja >= alvo:
                break
            if int(i) not in escolhidos:
                escolhidos.add(int(i))
                ja += 1
    return np.array(sorted(escolhidos), dtype=int)


def escolher_L(y: np.ndarray, fracao: float, seed: int, modo: str = "por_classe") -> np.ndarray:
    """`por_classe` (novo, por defeito) ou `raro` (regra antiga: classe positiva mais rara)."""
    if modo == "por_classe":
        return selecionar_L_por_classe(y, fracao, seed)
    if modo == "raro":
        return selecionar_L(y, fracao, seed)
    raise ValueError(modo)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pipeline de dados (PL08 §2.2) — CODEBRIM / Tema C.")
    ap.add_argument("--csv", default="data/splits.csv")
    ap.add_argument("--verificar", action="store_true", help="verifica fugas e imprime contagens")
    ap.add_argument("--gerar-json", action="store_true", help="escreve data/splits.json com hashes e L0")
    ap.add_argument("--determinismo", action="store_true", help="confirma o mesmo 1.o lote para a mesma seed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--L", type=float, default=0.05)
    args = ap.parse_args(argv)

    df = C.carregar(args.csv)
    print(f"splits: {args.csv}  ({len(df)} linhas)")
    verificar(df)
    utilizavel, vazios = excluir_sem_rotulo(df)
    print(f"\nrecortes utilizaveis: {len(utilizavel)}   excluidos (vetor todo a zero): {vazios}")
    contar(utilizavel)

    if args.gerar_json:
        import json

        pool = utilizavel[utilizavel.split == "train"].reset_index(drop=True)
        y = C.matriz_rotulos(pool)
        info = {
            "csv": str(args.csv),
            "hash_csv": C.hash_indices(range(len(df))),
            "recortes": int(len(utilizavel)),
            "excluidos_sem_rotulo": vazios,
            "por_split": {s: int((utilizavel.split == s).sum()) for s in ("train", "val", "test")},
            "grupos_por_split": {s: int(utilizavel[utilizavel.split == s].grupo.nunique()) for s in ("train", "val", "test")},
            "fugas": 0,
            "L_fracao": args.L,
            "seeds": {},
        }
        for seed in (0, 1, 2):
            idx = selecionar_L(y, args.L, seed)
            info["seeds"][str(seed)] = {
                "n": int(len(idx)),
                "hash": C.hash_indices(idx),
                "pos_por_classe": {c: int(y[idx, i].sum()) for i, c in enumerate(C.CLASSES)},
                "indices": idx.tolist(),
            }
            print(f"L seed={seed}: {len(idx)} recortes, hash {info['seeds'][str(seed)]['hash']}, "
                  f"eflorescencia={y[idx, C.CLASSES.index('Efflorescence')].sum():.0f}")
        C.escrever_json(info, C.DADOS / "splits.json")
        print(f"\nescrito {C.DADOS / 'splits.json'}")

    if args.determinismo:
        print("\n-- determinismo: 2 execucoes do 1.o lote de treino")
        lotes = []
        for _ in range(2):
            tr, _, _ = carregar_splits(args.csv, seed=args.seed, batch_size=16)
            x, y = next(iter(tr))
            lotes.append((x.sum().item(), y.sum().item(), tuple(y.shape)))
        print(f"   execucao 1: {lotes[0]}")
        print(f"   execucao 2: {lotes[1]}")
        assert lotes[0][0] == lotes[1][0], "o 1.o lote nao e reproduzivel com a mesma seed"
        print("   OK: identico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
