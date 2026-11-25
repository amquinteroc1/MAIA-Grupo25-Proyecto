"""
register_thesis_mlflow.py
Registra automáticamente en MLflow las 9 corridas oficiales de tesis (E0, E1, E2 × semillas 42, 123, 2026)
a partir de los resúmenes consolidados (outputs/kaggle_batch/**/batch_summary_E*.json) y las gráficas
de pruebas estadísticas de Wilcoxon, sin necesidad de reentrenar los notebooks en GPU.

Uso:
  python -m src.utils.register_thesis_mlflow
  python -m src.utils.register_thesis_mlflow --tracking-uri http://localhost:5000
  python -m src.utils.register_thesis_mlflow --tracking-uri https://sleep-mlflow.whitesky-80270a35.eastus.azurecontainerapps.io
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any

import mlflow
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_NAME = "sleep-edfx-tesis-E0-E1-E2"

ARCH_META: Dict[str, Dict[str, Any]] = {
    "E0": {
        "architecture": "SleepCNN1D",
        "temporal_context_epochs": 1,
        "temporal_window_s": 30,
        "description": "CNN 1D base sobre época individual de 30s (3x3000) sin contexto temporal.",
        "st_f1_macro_mean": 0.6090,
        "cross_domain_retention": 0.80,
    },
    "E1": {
        "architecture": "SleepCNN1D_E1_Concat",
        "temporal_context_epochs": 5,
        "temporal_window_s": 150,
        "description": "CNN 1D compartida sobre 5 épocas consecutivas con concatenación posicional fija.",
        "st_f1_macro_mean": 0.5893,
        "cross_domain_retention": 0.75,
    },
    "E2": {
        "architecture": "SleepCNN1D_E2_GRU",
        "temporal_context_epochs": 5,
        "temporal_window_s": 150,
        "description": "CNN 1D compartida + GRU sobre 5 épocas consecutivas (Mejor modelo de tesis).",
        "st_f1_macro_mean": 0.6354,
        "cross_domain_retention": 0.92,
    },
}


def find_summary_files() -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    base = ROOT_DIR / "outputs" / "kaggle_batch"
    for exp_id in ("E0", "E1", "E2"):
        matches = list(base.rglob(f"batch_summary_{exp_id}.json"))
        if matches:
            found[exp_id] = matches[0]
    return found


def register_runs(tracking_uri: str, replace: bool = False) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id_mlflow = client.create_experiment(
            EXPERIMENT_NAME,
            tags={
                "project": "MAIA Grupo 25 - Sleep-EDFX",
                "university": "Universidad de los Andes",
                "standard": "R&K / AASM",
            },
        )
    else:
        exp_id_mlflow = exp.experiment_id

    existing_runs = {
        r.info.run_name: r.info.run_id
        for r in client.search_runs(experiment_ids=[exp_id_mlflow])
    }

    summary_files = find_summary_files()
    if not summary_files:
        raise FileNotFoundError(
            f"No se encontraron archivos batch_summary_E*.json bajo {ROOT_DIR / 'outputs' / 'kaggle_batch'}."
        )

    wilcoxon_dir = ROOT_DIR / "outputs" / "metrics" / "wilcoxon"
    logged = 0
    skipped = 0

    for exp_code in ("E0", "E1", "E2"):
        summary_path = summary_files.get(exp_code)
        if not summary_path:
            print(f"[WARN] Falta batch_summary_{exp_code}.json, se omite {exp_code}.")
            continue

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        meta = ARCH_META[exp_code]
        ckpt_info = payload.get("checkpoint_verified") or {}

        for seed_str, seed_data in payload.get("seeds", {}).items():
            run_name = f"{exp_code}_seed{seed_str}"
            if run_name in existing_runs:
                if replace:
                    client.delete_run(existing_runs[run_name])
                else:
                    print(f"[SKIP] {run_name} ya existe en '{EXPERIMENT_NAME}'.")
                    skipped += 1
                    continue

            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=run_name):
                # Tags de trazabilidad clínica y académica
                mlflow.set_tags(
                    {
                        "experiment_code": exp_code,
                        "architecture": meta["architecture"],
                        "seed": str(seed_str),
                        "dataset": "Sleep-EDFX (SC-train / SC-val / SC-test / ST)",
                        "normalization": "SC-train Z-score (norm_params.npz)",
                        "source_repo": payload.get("source", "hf://maia-grupo25/sleep-edfx-models"),
                        "finished_at_utc": str(seed_data.get("finished_at_utc", "")),
                        "description": meta["description"],
                        "best_model_candidate": "true" if exp_code == "E2" else "false",
                    }
                )

                # Hiperparámetros de entrenamiento + tamaño de splits
                params: Dict[str, Any] = {
                    "experiment": exp_code,
                    "architecture": meta["architecture"],
                    "seed": seed_str,
                    "temporal_context_epochs": meta["temporal_context_epochs"],
                    "temporal_window_s": meta["temporal_window_s"],
                }
                for k, v in (seed_data.get("training_config") or {}).items():
                    params[k] = v
                for k, v in (seed_data.get("dataset_sizes") or {}).items():
                    params[f"n_epochs_{k}"] = v
                if ckpt_info and str(seed_str) == "42":
                    params["checkpoint_file"] = ckpt_info.get("file", "")
                    params["checkpoint_size_mb"] = ckpt_info.get("size_mb", "")
                mlflow.log_params({k: str(v) for k, v in params.items()})

                # Métricas oficiales de validación y SC-test + referencia ST
                metrics: Dict[str, float] = {
                    "best_val_f1_macro": float(seed_data.get("best_val_f1", 0.0)),
                    "st_f1_macro_arch_mean": float(meta["st_f1_macro_mean"]),
                    "cross_domain_retention": float(meta["cross_domain_retention"]),
                }
                for k, v in (seed_data.get("test_metrics") or {}).items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
                mlflow.log_metrics(metrics)

                # Artefactos (JSON resumen + figuras Wilcoxon en la semilla 42)
                mlflow.log_artifact(str(summary_path), artifact_path="summaries")
                if str(seed_str) == "42" and wilcoxon_dir.is_dir():
                    for fig in sorted(wilcoxon_dir.glob("*.png")):
                        mlflow.log_artifact(str(fig), artifact_path="wilcoxon_figures")

                logged += 1
                print(
                    f"[OK] Registrado {run_name} -> "
                    f"test_f1_macro={metrics.get('test_f1_macro', 0):.4f}, "
                    f"test_f1_n1={metrics.get('test_f1_n1', 0):.4f}, "
                    f"test_kappa={metrics.get('test_kappa', 0):.4f}"
                )

    print(
        f"\n[DONE] Experimento '{EXPERIMENT_NAME}' actualizado en {tracking_uri} "
        f"(nuevas corridas={logged}, omitidas={skipped})."
    )

    # Si registramos en SQLite local, normalizar rutas Windows -> /app/mlruns para Docker y Azure
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.replace("sqlite:///", "", 1))
        if db_path.is_file():
            prefix_posix = (ROOT_DIR / "mlruns").as_posix()
            con = sqlite3.connect(str(db_path))
            for old in (f"file:///{prefix_posix}", f"file:{prefix_posix}"):
                con.execute(
                    "UPDATE experiments SET artifact_location = REPLACE(artifact_location, ?, '/app/mlruns') WHERE name = ?",
                    (old, EXPERIMENT_NAME),
                )
                con.execute(
                    "UPDATE runs SET artifact_uri = REPLACE(artifact_uri, ?, '/app/mlruns') WHERE experiment_id = ?",
                    (old, exp_id_mlflow),
                )
            con.commit()
            con.close()
            print("[OK] Rutas de artefactos normalizadas a '/app/mlruns' para Docker y Azure.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Registra en MLflow las 9 corridas oficiales de tesis (E0, E1, E2)."
    )
    parser.add_argument(
        "--tracking-uri",
        default=f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}",
        help="URI de MLflow (por defecto sqlite:///mlflow.db local).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Reemplaza corridas existentes con el mismo nombre si ya fueron registradas.",
    )
    args = parser.parse_args()
    register_runs(tracking_uri=args.tracking_uri, replace=args.replace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
