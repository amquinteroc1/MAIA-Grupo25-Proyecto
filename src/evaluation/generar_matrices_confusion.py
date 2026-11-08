"""Matrices de confusion de los experimentos E0, E1 y E2.

Ubicacion prevista en el repositorio:
    src/evaluation/generar_matrices_confusion.py

Ejecucion desde la raiz del repositorio:
    python -m src.evaluation.generar_matrices_confusion

Lee exclusivamente predicciones ya guardadas; NO entrena ni vuelve a inferir.

Genera en src/evaluation/imagenes/matrices_confusion/:
    sc_test/     9 matrices por semilla (PNG + SVG) y tablas CSV
    st/          9 matrices por semilla (PNG + SVG) y tablas CSV
    resumen/     6 matrices promedio entre semillas (PNG + SVG)
                 2 figuras comparativas para el articulo (PNG + SVG)
                 6 CSV de matrices porcentuales promedio

Convencion: filas = clase real, columnas = clase predicha.
Porcentaje normalizado POR FILA (cada clase real suma 100%).
La matriz promedio es la media de los porcentajes de 3 semillas,
NO la suma de las 3 matrices como si fueran sujetos independientes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Compatible con Windows/PowerShell sin interfaz grafica
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix


CLASES = ("Wake", "N1", "N2", "N3", "REM")
LABELS = np.arange(len(CLASES), dtype=int)
MODELOS = ("E0", "E1", "E2")
SEMILLAS = (42, 123, 2026)
DOMINIOS = {"sc_test": ("test", 39961), "st": ("st", 42003)}
TITULOS = {"sc_test": "SC-test", "st": "ST"}
NOMBRES = {"E0": "E0 - CNN", "E1": "E1 - Concatenacion", "E2": "E2 - GRU"}


def cargar(folder: Path, prefijo: str, sufijo: str, modelo: str, semilla: int) -> np.ndarray:
    path = folder / f"{prefijo}_{sufijo}_{modelo}_seed{semilla}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Falta el archivo: {path}\n"
            "Comprueba que las predicciones se guardaron para este modelo y dominio."
        )
    return np.load(path, allow_pickle=False).reshape(-1)


def verificar_etiquetas(y: np.ndarray, contexto: str) -> None:
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{contexto}: las etiquetas no son enteras (dtype={y.dtype})")
    if not np.isin(y, LABELS).all():
        raise ValueError(f"{contexto}: hay etiquetas distintas de 0, 1, 2, 3 o 4")


def normalizar(cm: np.ndarray) -> np.ndarray:
    denominador = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm, denominador, out=np.zeros_like(cm, dtype=float),
        where=denominador != 0,
    ) * 100.0


def crear_ejes(ax, matriz: np.ndarray, titulo: str, conteos=None, cmap="Blues",
               mostrar_ejes=True, fontsize=10) -> None:
    im = ax.imshow(matriz, cmap=cmap, vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(5), CLASES)
    ax.set_yticks(range(5), CLASES)
    ax.tick_params(axis="both", labelsize=fontsize, length=0)
    ax.set_xlabel("Estadio predicho" if mostrar_ejes else "", fontsize=fontsize + 1, labelpad=8)
    ax.set_ylabel("Estadio real" if mostrar_ejes else "", fontsize=fontsize + 1, labelpad=8)
    ax.set_title(titulo, fontsize=fontsize + 2, pad=14, weight="bold")

    for i in range(5):
        for j in range(5):
            valor = float(matriz[i, j])
            if conteos is not None:
                etiqueta = f"{int(conteos[i, j]):,}\n({valor:.1f}%)".replace(",", ".")
            else:
                etiqueta = f"{valor:.1f}%"
            ax.text(j, i, etiqueta, ha="center", va="center",
                    fontsize=max(7, fontsize - 1),
                    color="white" if valor >= 54 else "#1d2d3d",
                    weight="semibold" if i == j else "normal")

    # Divisiones sutiles entre celdas, sin abarrotar el grafico.
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def guardar_grafica(fig, nombre: Path, dpi: int) -> None:
    nombre.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(nombre.with_suffix(".png"), dpi=dpi, bbox_inches="tight",
                pad_inches=0.13, facecolor="white")
    fig.savefig(nombre.with_suffix(".svg"), bbox_inches="tight",
                pad_inches=0.13, facecolor="white")
    plt.close(fig)


def guardar_csv_matriz(archivo: Path, matriz: np.ndarray, porcentajes: bool) -> None:
    archivo.parent.mkdir(parents=True, exist_ok=True)
    with archivo.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Real / predicho", *CLASES])
        for nombre, fila in zip(CLASES, matriz):
            if porcentajes:
                writer.writerow([nombre, *(f"{float(v):.4f}" for v in fila)])
            else:
                writer.writerow([nombre, *(int(v) for v in fila)])


def figura_individual(cm: np.ndarray, dominio: str, modelo: str, seed: int, destino: Path, dpi: int):
    figura, ax = plt.subplots(figsize=(7.0, 6.2))
    porcentajes = normalizar(cm)
    crear_ejes(ax, porcentajes, f"{NOMBRES[modelo]}  |  {TITULOS[dominio]}  |  semilla {seed}",
               conteos=cm, fontsize=11)
    figura.text(.50, .025,
                "Cada celda: numero de epocas (porcentaje sobre la clase real).",
                ha="center", fontsize=9, color="#555555")
    figura.tight_layout(rect=[0, .07, 1, 1])
    guardar_grafica(figura, destino, dpi)


def figura_media(promedio: np.ndarray, dominio: str, modelo: str, destino: Path, dpi: int):
    figura, ax = plt.subplots(figsize=(7.0, 6.2))
    crear_ejes(ax, promedio,
               f"{NOMBRES[modelo]}  |  {TITULOS[dominio]}  |  promedio de 3 semillas",
               fontsize=11)
    figura.text(.50, .025,
                "Media de matrices normalizadas por clase real; valores porcentuales.",
                ha="center", fontsize=9, color="#555555")
    figura.tight_layout(rect=[0, .07, 1, 1])
    guardar_grafica(figura, destino, dpi)


def figura_comparacion(promedios: dict, dominio: str, destino: Path, dpi: int):
    fig, ejes = plt.subplots(1, 3, figsize=(15.2, 5.4), constrained_layout=False)
    for ax, modelo in zip(ejes, MODELOS):
        crear_ejes(ax, promedios[dominio][modelo], NOMBRES[modelo], fontsize=10)
    fig.suptitle(f"Matrices de confusion normalizadas - {TITULOS[dominio]}",
                 fontsize=16, y=.985, weight="bold")
    fig.text(.5, .02,
             "Porcentajes respecto de cada clase real; media de las tres semillas.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.05, right=.98, top=.88, bottom=.13, wspace=.30)
    guardar_grafica(fig, destino, dpi)


def main():
    parser = argparse.ArgumentParser(description="Matrices de confusion SC-test y ST")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Raiz del repositorio (se detecta automaticamente)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Carpeta alternativa para las imagenes y CSV")
    parser.add_argument("--dpi", type=int, default=350,
                        help="Resolucion de PNG, predeterminado 350 dpi")
    args = parser.parse_args()

    root = args.root.resolve()
    base = (args.output_dir or root / "src" / "evaluation" / "imagenes" / "matrices_confusion").resolve()
    experimentos = root / "experiments"
    if not experimentos.is_dir():
        parser.error(f"No existe la carpeta de experimentos: {experimentos}")

    referencias = {}
    grupos_ref = {}
    promedios = {dominio: {} for dominio in DOMINIOS}
    resumen = []

    # Primero leer y validar todas las 18 evaluaciones, antes de crear archivos.
    matrices = {}
    for dominio, (sufijo, n_esperado) in DOMINIOS.items():
        for modelo in MODELOS:
            for seed in SEMILLAS:
                carpeta = experimentos / modelo / f"seed{seed}"
                ctx = f"{dominio} {modelo} semilla {seed}"
                y_true = cargar(carpeta, "y_true", sufijo, modelo, seed)
                y_pred = cargar(carpeta, "y_pred", sufijo, modelo, seed)
                grupos = cargar(carpeta, "groups", sufijo, modelo, seed)
                if len(y_true) != n_esperado or len(y_pred) != n_esperado or len(grupos) != n_esperado:
                    raise ValueError(f"{ctx}: longitud incorrecta (esperadas {n_esperado} epocas)")
                verificar_etiquetas(y_true, ctx + " y_true")
                verificar_etiquetas(y_pred, ctx + " y_pred")

                if dominio not in referencias:
                    referencias[dominio] = y_true
                    grupos_ref[dominio] = grupos
                elif (not np.array_equal(y_true, referencias[dominio]) or
                      not np.array_equal(grupos, grupos_ref[dominio])):
                    raise ValueError(f"{ctx}: la cohorte no coincide con las demas semillas/modelos")

                matrices[(dominio, modelo, seed)] = confusion_matrix(
                    y_true, y_pred, labels=LABELS)

    if len(np.unique(grupos_ref["st"])) != 22:
        raise ValueError("ST debe incluir exactamente 22 sujetos")

    # Las matrices de porcentajes se promedian, sin duplicar observaciones.
    for dominio in DOMINIOS:
        for modelo in MODELOS:
            normalizadas = []
            for seed in SEMILLAS:
                cm = matrices[(dominio, modelo, seed)]
                pct = normalizar(cm)
                normalizadas.append(pct)
                nombre = f"matriz_{dominio}_{modelo}_seed{seed}"
                dest = base / dominio / nombre
                figura_individual(cm, dominio, modelo, seed, dest, args.dpi)
                guardar_csv_matriz(dest.with_name(nombre + "_conteos.csv"), cm, porcentajes=False)
                guardar_csv_matriz(dest.with_name(nombre + "_porcentajes.csv"), pct, porcentajes=True)
                resumen.append((dominio, modelo, seed, len(referencias[dominio]),
                                len(np.unique(grupos_ref[dominio]))))
                print(f"Generada: {dominio} {modelo}, semilla {seed}")

            promedio = np.mean(np.stack(normalizadas), axis=0)
            promedios[dominio][modelo] = promedio
            nombre = f"matriz_promedio_{dominio}_{modelo}"
            destino = base / "resumen" / nombre
            figura_media(promedio, dominio, modelo, destino, args.dpi)
            guardar_csv_matriz(destino.with_name(nombre + "_porcentajes.csv"),
                               promedio, porcentajes=True)

        figura_comparacion(
            promedios, dominio,
            base / "resumen" / f"comparativa_{dominio}_E0_E1_E2",
            args.dpi,
        )

    resumen_csv = base / "resumen" / "cohortes_evaluadas.csv"
    with resumen_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dominio", "modelo", "semilla", "epocas", "sujetos"])
        writer.writerows(resumen)

    print("\nVERIFICACION OK: 18 matrices, mismas etiquetas y orden de sujetos por dominio.")
    print(f"SC-test: {len(referencias['sc_test']):,} epocas, {len(np.unique(grupos_ref['sc_test']))} sujetos")
    print(f"ST: {len(referencias['st']):,} epocas, {len(np.unique(grupos_ref['st']))} sujetos")
    print(f"Imagenes PNG/SVG y tablas CSV guardadas en: {base}")
    print("Figuras recomendadas para el articulo:")
    print(f"  {base / 'resumen' / 'comparativa_sc_test_E0_E1_E2.png'}")
    print(f"  {base / 'resumen' / 'comparativa_st_E0_E1_E2.png'}")


if __name__ == "__main__":
    main()
