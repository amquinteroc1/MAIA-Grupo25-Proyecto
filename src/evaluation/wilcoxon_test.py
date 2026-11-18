#!/usr/bin/env python3
"""Wilcoxon pareado por persona para E0, E1 y E2 (Sleep-EDF SC-test y ST).

Requiere solamente predicciones y grupos guardados al finalizar cada entrenamiento.
No carga X, no reentrena, no necesita GPU ni EDF originales.

Uso:
  python wilcoxon_tesis.py --experiments-dir experiments
  python wilcoxon_tesis.py --experiments-dir _hf_wilcoxon/tesis-final-2026/experiments

Comparaciones confirmatorias: F1-macro, 6 contrastes con correccion Holm.
Exploratorias: F1 de N1, 6 contrastes con correccion Holm separada.
Unidad estadistica: SUJETO. Las tres semillas se promedian dentro de sujeto.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon
from sklearn.metrics import f1_score

MODELOS = ("E0", "E1", "E2")
SEMILLAS = (42, 123, 2026)
DOMINIOS = {"SC-test": ("test", 39961, 16), "ST": ("st", 42003, 22)}
PARES = (("E1", "E0"), ("E2", "E0"), ("E2", "E1"))
METRICAS = ("f1_macro", "f1_n1")
CLASES = np.arange(5, dtype=np.int64)


def load(experiments: Path, modelo: str, semilla: int, sufijo: str, tipo: str) -> np.ndarray:
    path = experiments / modelo / f"seed{semilla}" / f"{tipo}_{sufijo}_{modelo}_seed{semilla}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}. Descarga los archivos 'experiments/**' del PR #4.")
    a = np.load(path, allow_pickle=False)
    return a.reshape(-1)


def load_metrics(experiments: Path) -> pd.DataFrame:
    out = []
    for dominio, (sufijo, n_epocas, n_sujetos) in DOMINIOS.items():
        referencia_y = referencia_g = None
        for modelo in MODELOS:
            for semilla in SEMILLAS:
                y = load(experiments, modelo, semilla, sufijo, "y_true")
                pred = load(experiments, modelo, semilla, sufijo, "y_pred")
                grupos = load(experiments, modelo, semilla, sufijo, "groups")

                if not (len(y) == len(pred) == len(grupos) == n_epocas):
                    raise ValueError(f"Tamanos inconsistentes: {dominio}, {modelo}, {semilla} "
                                     f"(esperado {n_epocas}; visto {len(y)}, {len(pred)}, {len(grupos)})")
                if not (np.issubdtype(y.dtype, np.integer) and
                        np.issubdtype(pred.dtype, np.integer)):
                    raise ValueError(f"Etiquetas no enteras: {dominio}, {modelo}, {semilla}")
                if not (np.isin(y, CLASES).all() and np.isin(pred, CLASES).all()):
                    raise ValueError(f"Etiquetas fuera de 0..4: {dominio}, {modelo}, {semilla}")
                if len(np.unique(grupos)) != n_sujetos:
                    raise ValueError(f"Numero inesperado de personas en {dominio} "
                                     f"{modelo}/seed{semilla}: {len(np.unique(grupos))} != {n_sujetos}")
                if referencia_y is None:
                    referencia_y, referencia_g = y.copy(), grupos.copy()
                elif not (np.array_equal(y, referencia_y) and np.array_equal(grupos, referencia_g)):
                    raise ValueError(f"Cohorte, orden, etiquetas o IDs no coinciden: "
                                     f"{dominio} {modelo}/seed{semilla}. No se puede parear.")

                for sujeto in np.unique(grupos):
                    mask = grupos == sujeto
                    yi, pi = y[mask], pred[mask]
                    n_n1 = int((yi == 1).sum())
                    f1_macro = f1_score(yi, pi, labels=CLASES,
                                        average="macro", zero_division=0)
                    f1_n1 = (float(f1_score(yi, pi, labels=[1],
                                            average=None, zero_division=0)[0])
                             if n_n1 > 0 else np.nan)
                    out.append({"dominio": dominio,
                                "sujeto": str(sujeto), "modelo": modelo,
                                "semilla": int(semilla),
                                "epocas_sujeto": int(mask.sum()),
                                "n_n1": n_n1,
                                "f1_macro": float(f1_macro),
                                "f1_n1": f1_n1})
        print(f"Cohorte validada {dominio}: {n_epocas:,} epocas, "
              f"{n_sujetos} personas; orden y etiquetas identicos entre 9 modelos.")
    return pd.DataFrame(out)


def holm(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    result = np.empty(m, dtype=float)
    previous = 0.0
    for j, pos in enumerate(order):
        previous = max(previous, (m - j) * p_values[pos])
        result[pos] = min(1.0, previous)
    return result


def plot_differences(df: pd.DataFrame, metrica: str, dominio: str, output: Path) -> None:
    sub = df[(df["metrica"] == metrica) & (df["dominio"] == dominio)]
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(2026)
    for i, par in enumerate(PARES):
        nombre = f"{par[0]} - {par[1]}"
        d = sub[sub["comparacion"] == nombre]["delta"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(i + jitter, d, s=28, alpha=0.72, label=None)
        ax.hlines(np.median(d), i - 0.22, i + 0.22, linewidth=2.5, color="black")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="grey")
    ax.set_xticks(np.arange(3), [f"{a} - {b}" for a, b in PARES])
    ax.set_ylabel("Diferencia pareada de F1 (promedio sobre tres semillas)")
    ax.set_title(f"Wilcoxon por persona: {dominio}, {metrica}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyse(df: pd.DataFrame, outdir: Path, n_bootstrap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    promedios = df.groupby(["dominio", "sujeto", "modelo"], as_index=False)[list(METRICAS)].mean()
    promedios.to_csv(outdir / "metricas_promedio_por_sujeto.csv", index=False)
    descriptivos = promedios.groupby(["dominio", "modelo"], as_index=False).agg(
        n_sujetos=("sujeto", "nunique"),
        f1_macro_media=("f1_macro", "mean"),
        f1_macro_mediana=("f1_macro", "median"),
        f1_macro_sd=("f1_macro", "std"),
        f1_n1_media=("f1_n1", "mean"),
        sujetos_con_n1=("f1_n1", "count"),
    )
    descriptivos.to_csv(outdir / "resumen_descriptivo_sujetos.csv", index=False)

    resultados = []
    diferencias = []
    rng = np.random.default_rng(2026)

    for dominio in DOMINIOS:
        datos = promedios[promedios["dominio"] == dominio]
        for metrica in METRICAS:
            pivot = datos.pivot(index="sujeto", columns="modelo", values=metrica)
            for modelo_a, modelo_b in PARES:
                paired = pivot[[modelo_a, modelo_b]].dropna()
                delta = np.round((paired[modelo_a] - paired[modelo_b]).to_numpy(dtype=float), 12)
                sujetos = paired.index.to_numpy()
                if len(delta) < 2:
                    raise ValueError(f"Menos de dos sujetos para {dominio}/{metrica}/{modelo_a}-{modelo_b}")

                for sujeto, valor, a, b in zip(sujetos, delta,
                                                paired[modelo_a].to_numpy(),
                                                paired[modelo_b].to_numpy()):
                    diferencias.append({"dominio": dominio, "metrica": metrica,
                                        "comparacion": f"{modelo_a} - {modelo_b}",
                                        "sujeto": sujeto, "f1_A": a, "f1_B": b,
                                        "delta": valor})

                nz = delta[delta != 0]
                if len(nz):
                    resultado = wilcoxon(delta, alternative="two-sided",
                                         zero_method="wilcox", method="auto")
                    w, p = float(resultado.statistic), float(resultado.pvalue)
                    ranks = rankdata(np.abs(nz), method="average")
                    r_rb = float((np.sum(ranks[nz > 0]) - np.sum(ranks[nz < 0]))
                                 / np.sum(ranks))
                    p_signo = float(binomtest(int(np.sum(nz > 0)),
                                                  len(nz), p=0.5).pvalue)
                else:
                    w, p, r_rb, p_signo = 0.0, 1.0, 0.0, 1.0

                # Percentile bootstrap: IC DESCRIPTIVO de mediana, no IC exacto de Wilcoxon.
                med_boot = np.median(rng.choice(delta, size=(n_bootstrap, len(delta)),
                                               replace=True), axis=1)
                lo, hi = np.percentile(med_boot, [2.5, 97.5])
                resultados.append({
                    "dominio": dominio, "metrica": metrica,
                    "comparacion": f"{modelo_a} - {modelo_b}",
                    "n_sujetos": len(delta), "n_no_cero": len(nz),
                    "sujetos_mejoran": int(np.sum(delta > 0)),
                    "sujetos_empeoran": int(np.sum(delta < 0)),
                    "sujetos_empatan": int(np.sum(delta == 0)),
                    "media_delta": float(np.mean(delta)),
                    "mediana_delta": float(np.median(delta)),
                    "ic95_boot_mediana_inf": float(lo),
                    "ic95_boot_mediana_sup": float(hi),
                    "W": w, "p_wilcoxon": p,
                    "r_biserial_rangos": r_rb,
                    "p_signos_sin_ajustar": p_signo,
                })

    res = pd.DataFrame(resultados)
    detail = pd.DataFrame(diferencias)
    for metrica in METRICAS:
        indices = res.index[res["metrica"] == metrica]
        res.loc[indices, "p_holm"] = holm(res.loc[indices, "p_wilcoxon"].to_numpy())
    res["significativo_005"] = res["p_holm"] < 0.05
    res["familia_holm"] = "6 contrastes de la misma metrica (3 pares x 2 dominios)"
    res.to_csv(outdir / "resultados_wilcoxon.csv", index=False)
    detail.to_csv(outdir / "diferencias_pareadas_por_sujeto.csv", index=False)
    for dominio in DOMINIOS:
        for metrica in METRICAS:
            plot_differences(detail, metrica, dominio,
                             outdir / f"diferencias_{dominio.lower().replace('-', '_')}_{metrica}.png")

    print("\nRESUMEN DESCRIPTIVO POR PERSONA (no confundir con F1 global):")
    print(descriptivos.round(4).to_string(index=False))
    print("\nWILCOXON - RESULTADOS:")
    print(res[["dominio", "metrica", "comparacion", "n_sujetos",
               "mediana_delta", "W", "p_wilcoxon", "p_holm",
               "r_biserial_rangos", "significativo_005"]].round(5).to_string(index=False))
    return res, descriptivos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/metrics/wilcoxon"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        parser.error("--n-bootstrap debe ser >= 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.experiments_dir)
    df.to_csv(args.output_dir / "metricas_por_sujeto_semilla.csv", index=False)
    analyse(df, args.output_dir, args.n_bootstrap)
    metadata = {
        "origen": str(args.experiments_dir.resolve()),
        "dominios": {d: {"epocas": a[1], "sujetos": a[2]} for d, a in DOMINIOS.items()},
        "modelos": list(MODELOS), "semillas": list(SEMILLAS),
        "unidad_inferencial": "persona, promedio F1 de 3 semillas por arquitectura",
        "cohorte": "las mismas etiquetas y grupos en los nueve pares modelo-semilla",
        "metrica_principal": "F1-macro con labels=0..4; clases ausentes tienen F1=0",
        "metrica_secundaria": "F1-N1, NA cuando un sujeto no presenta N1 real",
        "correccion": "Holm sobre 6 contrastes de F1-macro y separadamente 6 de F1-N1",
        "bootstrap": "intervalo percentil descriptivo para mediana de diferencia, 10000 remuestreos por defecto",
        "observacion": "Wilcoxon asume simetria de diferencias para interpretacion de localizacion; inspeccionar graficas",
    }
    (args.output_dir / "metodologia.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArchivos y graficos guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
