"""Evaluacion externa de E0, E1 y E2 sobre Sleep Telemetry (ST).

Ejecutar desde la raiz del repositorio:
    python -m src.evaluation.evaluacion_st

Requiere los archivos *_st_*.npy generados por el entrenamiento.
"""
from pathlib import Path
import csv
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
)

ROOT = Path(__file__).resolve().parents[2]  # raiz: MAIA-Grupo25-Proyecto
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "outputs" / "metrics"
N_EXPECTED = 42003


def main():
    reference_y = None
    reference_groups = None
    results = []

    for experiment in ("E0", "E1", "E2"):
        for seed in (42, 123, 2026):
            folder = EXPERIMENTS / experiment / f"seed{seed}"

            def load(prefix):
                file_path = folder / f"{prefix}_{experiment}_seed{seed}.npy"
                if not file_path.is_file():
                    raise FileNotFoundError(f"Falta archivo: {file_path}")
                return np.load(file_path, allow_pickle=False)

            y_true = load("y_true_st")
            y_pred = load("y_pred_st")
            groups = load("groups_st")

            if not (len(y_true) == len(y_pred) == len(groups) == N_EXPECTED):
                raise ValueError(f"Tamano ST incorrecto en {experiment}, semilla {seed}")
            if reference_y is None:
                reference_y = y_true
                reference_groups = groups
            elif not (np.array_equal(y_true, reference_y) and
                      np.array_equal(groups, reference_groups)):
                raise ValueError(f"Cohorte ST no coincide en {experiment}, semilla {seed}")

            result = {
                "modelo": experiment,
                "semilla": seed,
                "epocas": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0),
                "f1_n1": f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0],
                "kappa": cohen_kappa_score(y_true, y_pred),
            }
            results.append(result)

    print("\nEVALUACION EXTERNA ST - 42003 EPOCAS")
    print("Modelo Semilla Accuracy BalAcc F1-macro F1-N1 Kappa")
    for row in results:
        print(f"{row['modelo']:6} {row['semilla']:7} "
              f"{row['accuracy']:.4f}   {row['balanced_accuracy']:.4f}  "
              f"{row['f1_macro']:.4f}   {row['f1_n1']:.4f} {row['kappa']:.4f}")

    print("\nRESUMEN POR MODELO (MEDIA +/- DESVIACION ESTANDAR)")
    for experiment in ("E0", "E1", "E2"):
        rows = [r for r in results if r["modelo"] == experiment]
        for metric in ("accuracy", "balanced_accuracy", "f1_macro", "f1_n1", "kappa"):
            vals = np.array([r[metric] for r in rows])
            print(f"{experiment} {metric}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
    print(f"\nSujetos/grupos ST: {len(np.unique(reference_groups))}")
    print("VERIFICACION DE COHORTE ST: CORRECTA")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "evaluacion_st_resumen.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV guardado en: {path}")


if __name__ == "__main__":
    main()
