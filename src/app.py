"""Assistente de anotacao (P3-D1..D3) — dono: A4 (Pedro).

O ponto do desenho, que vem da pesquisa do A4: **nao mostrar a previsao ao anotador
por omissao**. Expor a classe prevista ancora a decisao (Rosbach et al., 2026:
7% das decisoes corretas foram revertidas por sugestao errada da IA, em patologia).
A incerteza serve para ESCOLHER o que anotar; mostrar a resposta e outra decisao,
que se testa (`--revelar`) em vez de se assumir.

Custo de anotacao: o que se mede aqui e **tempo observado por recorte na app**, nao
"segundos por rotulo" estimados. Sem campanha real, o relatorio diz que e um piloto
com N anotacoes — nao um estudo de tempo.

Uso:
  python src/app.py --fila 40            # gera runs/fila_anotacao.csv (incerteza, sem revelar)
  python src/app.py --servir             # abre a app Gradio (requer `pip install gradio`)
  python src/app.py --resumo             # le runs/anotacoes.csv e resume custo/acordo
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comum as C  # noqa: E402

FILA = C.RUNS / "fila_anotacao.csv"
ANOT = C.RUNS / "anotacoes.csv"

# Nomes em português para a interface do anotador
NOMES_CLASSES_PT = {
    "Background": "Fundo (sem defeito)",
    "CorrosionStain": "Mancha de Corrosão",
    "Crack": "Fissura",
    "Efflorescence": "Eflorescência",
    "ExposedBars": "Armadura Exposta",
    "Spallation": "Delaminação",
}


def gerar_fila(n: int = 40, seed: int = 0, estrategia: str = "entropia") -> Path:
    """Ordena o pool nao rotulado pela incerteza do modelo treinado com L (5%)."""
    import probe
    from baseline import HIPER, carregar_caracteristicas
    from dados import selecionar_L

    F, Y, df = carregar_caracteristicas()
    pool = np.flatnonzero(df.split.values == "train")
    lab = pool[selecionar_L(Y[pool], 0.05, seed)]
    m = probe.treinar(F, Y, lab, seed=seed, **HIPER)
    U = np.setdiff1d(pool, lab)
    P = probe.probabilidades(m, F[U])
    if estrategia == "entropia":
        score = probe.incerteza_entropia(P)
    elif estrategia == "margem":
        score = -probe.incerteza_margem(P)
    else:
        raise ValueError(estrategia)
    ordem = np.argsort(-score)[:n]
    fila = pd.DataFrame(
        {
            "ordem": np.arange(1, len(ordem) + 1),
            "path": df.path.values[U[ordem]],
            "grupo": df.grupo.values[U[ordem]],
            "incerteza": np.round(score[ordem], 4),
            "previsto_oculto": ["|".join(c for c, p in zip(C.CLASSES, P[i]) if p >= 0.5) for i in ordem],
            "anotado": 0,
        }
    )
    C.garantir_dirs()
    fila.to_csv(FILA, index=False)
    print(f"fila: {FILA}  ({len(fila)} recortes, estrategia={estrategia}, seed={seed})")
    print(f"  previsto_oculto so serve para AUDITORIA — a app nao o mostra ao anotador.")
    print(f"  grupos distintos na fila: {fila.grupo.nunique()} (se < {len(fila)}, ha quase-duplicados)")
    return FILA


def servir(revelar: bool = False, anotador: str = "anon", share: bool = False):
    """Interface gráfica em Gradio: recorte, caixas de seleção em português, gravar e cronómetro."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio não está instalado. Execute `pip install gradio`.")
        return 1
    if not FILA.exists():
        gerar_fila()
    fila = pd.read_csv(FILA)
    estado = {"i": 0, "t0": time.time()}

    def carregar():
        i = estado["i"]
        if i >= len(fila):
            return (None,) + tuple(False for _ in C.CLASSES) + ("### 🎉 Fila terminada!\nTodos os recortes desta ronda foram anotados.",)
        r = fila.iloc[i]
        estado["t0"] = time.time()
        nota = f"**Recorte {r.ordem} de {len(fila)}** — Incerteza do modelo: `{r.incerteza:.3f}` *(maior = mais prioritário)*"
        if revelar:
            sugestoes = [NOMES_CLASSES_PT.get(c, c) for c in (r.previsto_oculto.split("|") if pd.notna(r.previsto_oculto) and r.previsto_oculto else [])]
            nota += f"\n\n🤖 **Sugestão do modelo (condição com revelação):** {', '.join(sugestoes) if sugestoes else 'Nenhuma'}"
        return (str(C.caminho_imagem(r.path)),) + tuple(False for _ in C.CLASSES) + (nota,)

    def gravar(*vals):
        rotulos, i = vals[:-1], estado["i"]
        if i >= len(fila):
            return "Fila já terminada."
        r = fila.iloc[i]
        dt = time.time() - estado["t0"]
        linha = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ordem": int(r.ordem),
            "path": r.path,
            "grupo": r.grupo,
            "rotulos": "|".join(c for c, v in zip(C.CLASSES, rotulos) if v),
            "segundos": round(dt, 1),
            "revelado": bool(revelar),
            "anotador": anotador,
        }
        pd.DataFrame([linha]).to_csv(ANOT, mode="a", header=not ANOT.exists(), index=False)
        fila.loc[i, "anotado"] = 1
        fila.to_csv(FILA, index=False)
        estado["i"] += 1
        return f"✅ Anotação guardada em {dt:.1f}s — a carregar próximo recorte..."

    with gr.Blocks(title="CODEBRIM — Assistente de Anotação (Active Learning)") as app:
        gr.Markdown(
            "## 🔍 Assistente de Anotação Ativa — CODEBRIM (Tema C)\n"
            "Assinale todas as patologias visíveis no recorte de betão. "
            "Por omissão, o modelo **não exibe a sua previsão** para evitar enviesamento e ancoragem cognitiva (*Rosbach et al., 2026*)."
        )
        with gr.Row():
            img = gr.Image(type="filepath", label="Recorte de Inspeção de Betão", height=340)
            with gr.Column():
                gr.Markdown("### Patologias a Identificar:")
                caixas = [gr.Checkbox(label=NOMES_CLASSES_PT.get(c, c)) for c in C.CLASSES]
                nota = gr.Markdown("")
        with gr.Row():
            b_grav = gr.Button("💾 Gravar e Próximo", variant="primary", scale=2)
            b_ver = gr.Button("🔄 Recarregar Imagem (repor cronómetro)", scale=1)
        b_ver.click(lambda: carregar(), outputs=[img] + caixas + [nota])
        b_grav.click(gravar, inputs=caixas, outputs=[nota]).then(carregar, outputs=[img] + caixas + [nota])
        app.load(carregar, outputs=[img] + caixas + [nota])
    app.launch(share=share)
    return 0


def resumo() -> int:
    if not ANOT.exists():
        print(f"sem {ANOT} — ainda nao ha anotacoes registadas")
        return 1
    df = pd.read_csv(ANOT)
    print(f"anotacoes: {len(df)}  recortes distintos: {df.path.nunique()}  anotador(es): {df.anotador.unique().tolist()}")
    print(f"tempo por recorte: mediana {df.segundos.median():.1f}s  media {df.segundos.mean():.1f}s  "
          f"p90 {df.segundos.quantile(0.9):.1f}s")
    print(f"rotulos marcados por recorte: {(df.rotulos.fillna('').str.count(chr(124)) + 1).mean():.2f}")
    if df.path.duplicated().any():
        a = df.drop_duplicates("path", keep="first").set_index("path").rotulos
        b = df.drop_duplicates("path", keep="last").set_index("path").rotulos
        comuns = a.index.intersection(b.index)
        if len(comuns):
            iguais = sum(a[c] == b[c] for c in comuns)
            print(f"acordo bruto em {len(comuns)} recortes repetidos: {iguais / len(comuns):.1%} "
                  f"(kappa so com 2 anotadores independentes)")
        else:
            print("sem recortes anotados duas vezes: nao ha acordo a medir (nao se inventa kappa)")
    print(f"\nnota para o relatorio: isto e um piloto com {len(df)} anotacoes, nao um estudo de tempo.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fila", type=int, default=0, help="gera a fila com N recortes mais incertos")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--estrategia", default="entropia", choices=["entropia", "margem"])
    ap.add_argument("--servir", action="store_true")
    ap.add_argument("--share", action="store_true", help="cria link público partilhável do Gradio")
    ap.add_argument("--revelar", action="store_true", help="mostra a sugestao do modelo (condicao de teste de ancoragem)")
    ap.add_argument("--anotador", default="anon")
    ap.add_argument("--resumo", action="store_true")
    a = ap.parse_args()
    if a.fila:
        gerar_fila(a.fila, a.seed, a.estrategia)
    elif a.servir:
        servir(a.revelar, a.anotador, a.share)
    elif a.resumo:
        resumo()
    else:
        # Por omissão se for corrido diretamente, inicia a app
        servir(a.revelar, a.anotador, a.share)
