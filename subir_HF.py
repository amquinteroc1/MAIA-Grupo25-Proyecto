"""Sube artefactos de tesis E0/E1/E2 a dos repositorios Hugging Face.

Instalar: python -m pip install -U huggingface_hub hf_xet
Autenticar: hf auth login
Ejecutar desde raiz del repositorio:
    python -m src.evaluation.subir_huggingface_tesis --dry-run
    python -m src.evaluation.subir_huggingface_tesis

Deja intactos los archivos anteriores creando una nueva carpeta remota.
No sube EDF brutos, tokens, .env, Git, ni datos procesados grandes salvo opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RUNS = [(e, s) for e in ("E0", "E1", "E2") for s in (42, 123, 2026)]
DATA_REPO = "maia-grupo25/sleep-edfx-data"
WEIGHTS_REPO = "maia-grupo25/sleep-edfx-models"
DEFAULT_FOLDER = "revision-luigi"
DATA_EXT = {
    ".npy", ".npz", ".csv", ".tsv", ".json", ".jsonl", ".yaml",
    ".yml", ".md", ".txt", ".png", ".jpg", ".jpeg", ".svg",
    ".pdf", ".parquet", ".feather", ".h5", ".hdf5", ".zarr",
}
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
OMIT_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "node_modules", ".cache"}


def git_info(root: Path) -> dict:
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = None
        info["dirty"] = None
    return info


def iter_files(folder: Path, allowed: set[str]):
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not any(p in OMIT_DIRS for p in path.relative_to(folder).parts):
            if path.suffix.lower() in allowed:
                yield path


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"Destino duplicado en paquete: {dst}")
    try:
        os.link(src, dst)  # enlace duro en la misma unidad NTFS (sin duplicar GB)
    except OSError:
        shutil.copy2(src, dst)


def collect(root: Path, stage_data: Path, stage_models: Path, include_processed: bool,
            include_mlflow: bool):
    manifest = {"data": [], "models": [], "warnings": []}
    missing = []

    def add(src: Path, dst_base: Path, remote_relative: Path, category: str):
        link_or_copy(src, dst_base / remote_relative)
        manifest[category].append({"ruta": remote_relative.as_posix(), "bytes": src.stat().st_size})

    for experiment, seed in RUNS:
        run = root / "experiments" / experiment / f"seed{seed}"
        if not run.is_dir():
            missing.append(str(run))
            continue
        required = [
            f"y_true_test_{experiment}_seed{seed}.npy",
            f"y_pred_test_{experiment}_seed{seed}.npy",
            f"y_proba_test_{experiment}_seed{seed}.npy",
            f"groups_test_{experiment}_seed{seed}.npy",
            f"y_true_st_{experiment}_seed{seed}.npy",
            f"y_pred_st_{experiment}_seed{seed}.npy",
            f"y_proba_st_{experiment}_seed{seed}.npy",
            f"groups_st_{experiment}_seed{seed}.npy",
        ]
        missing.extend(str(run / name) for name in required if not (run / name).is_file())
        checkpoints = list(iter_files(run, WEIGHT_EXT))
        if not checkpoints:
            missing.append(f"{run} (checkpoint .pt/.safetensors/.ckpt no encontrado)")
        for src in iter_files(run, DATA_EXT):
            rel = Path("experiments") / experiment / f"seed{seed}" / src.relative_to(run)
            add(src, stage_data, rel, "data")
        for src in checkpoints:
            rel = Path("experiments") / experiment / f"seed{seed}" / src.relative_to(run)
            add(src, stage_models, rel, "models")

    for name in ("metrics", "predictions", "figures", "plots", "graficos", "reports"):
        folder = root / "outputs" / name
        for src in iter_files(folder, DATA_EXT):
            add(src, stage_data, Path("outputs") / name / src.relative_to(folder), "data")
    folder = root / "src" / "evaluation" / "imagenes"
    for src in iter_files(folder, DATA_EXT):
        add(src, stage_data, Path("src/evaluation/imagenes") / src.relative_to(folder), "data")

    if include_processed:
        for folder in (root / "outputs" / "data", root / "data" / "processed"):
            for src in iter_files(folder, DATA_EXT | WEIGHT_EXT):
                # Un tensor de datos .pt debe ir al repo de datos, no al de pesos
                rel = folder.relative_to(root) / src.relative_to(folder)
                add(src, stage_data, rel, "data")

    if include_mlflow:
        for name in ("mlruns.db", "mlflow.db"):
            src = root / name
            if src.is_file():
                add(src, stage_data, Path("tracking") / src.name, "data")
        manifest["warnings"].append(
            "La base MLflow puede registrar rutas locales o metadatos; revisar antes de publicarla."
        )

    return manifest, missing


def make_metadata(root: Path, stage_data: Path, stage_models: Path, manifest: dict):
    stamp = datetime.now(timezone.utc).isoformat()
    meta = {
        "fecha_utc": stamp,
        "proyecto": "MAIA Grupo 25: contexto temporal en estadios del sueno",
        "n_corridas": len(RUNS),
        "corridas": [{"experimento": e, "semilla": s} for e, s in RUNS],
        "sc_test_epocas": 39961,
        "st_epocas": 42003,
        "git": git_info(root),
        "ficheros": manifest,
        "notas": "SC-test y ST son cohortes distintas; ST se uso solo en evaluacion externa.",
    }
    for target, what in ((stage_data, "datos y evaluaciones"), (stage_models, "checkpoints y pesos")):
        (target / "MANIFEST.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (target / "README.md").write_text(
            f"# MAIA Grupo 25 - {what}\n\n"
            "Experimentos E0 (sin contexto), E1 (concatenacion de cinco epocas) y "
            "E2 (cinco epocas con GRU). Tres semillas: 42, 123, 2026.\n\n"
            "SC-test: 39.961 epocas. ST: 42.003 epocas de 22 sujetos, "
            "sin reentrenamiento.\n\n"
            "Los archivos originales se almacenan bajo `experiments/E*/seed*/`. "
            "La correspondencia, los tamanos y el commit estan en MANIFEST.json.\n",
            encoding="utf-8",
        )


def repo_type(api, repo_id: str):
    errors = []
    for kind in ("dataset", "model"):
        try:
            api.repo_info(repo_id=repo_id, repo_type=kind)
            return kind
        except Exception as exc:
            errors.append(f"{kind}: {exc.__class__.__name__}")
    raise RuntimeError(
        f"No se puede acceder a {repo_id}. Revisar nombre del repo, tipo, "
        f"token y permisos de escritura. Intentos: {errors}"
    )


def main():
    p = argparse.ArgumentParser(description="Subida segura de resultados E0/E1/E2 al Hub")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                   help="Raiz del repositorio local (por defecto, dos niveles sobre src/evaluation)")
    p.add_argument("--folder", default=DEFAULT_FOLDER, help="Nueva carpeta en ambos repositorios HF")
    p.add_argument("--branch", default="main", help="Rama HF; puede ser nueva")
    p.add_argument("--data-repo", default=DATA_REPO)
    p.add_argument("--weights-repo", default=WEIGHTS_REPO)
    p.add_argument("--dry-run", action="store_true", help="Inventario y validacion sin subir nada")
    p.add_argument("--include-processed-data", action="store_true",
                   help="Incluir tensores ya procesados de outputs/data y data/processed (pueden ser GB)")
    p.add_argument("--include-mlflow", action="store_true", help="Incluir SQLite MLflow (revisar metadatos)")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "experiments").is_dir():
        p.error(f"No existe {root / 'experiments'}. Ejecuta desde el proyecto o usa --root")
    if not args.folder or args.folder in {".", ".."} or "/" in args.folder or "\\" in args.folder:
        p.error("--folder debe ser un nombre de carpeta simple, sin barras")

    # Directorio temporal en la raiz: enlaces duros evitan duplicar archivos grandes en NTFS.
    with tempfile.TemporaryDirectory(prefix="_hf_stage_", dir=str(root)) as temp:
        data_dir = Path(temp) / "data"
        models_dir = Path(temp) / "models"
        data_dir.mkdir()
        models_dir.mkdir()
        manifest, missing = collect(root, data_dir, models_dir,
                                   args.include_processed_data, args.include_mlflow)
        if missing:
            print("ERROR: faltan archivos necesarios. No se subira nada:", file=sys.stderr)
            for item in missing:
                print("  -", item, file=sys.stderr)
            return 2
        make_metadata(root, data_dir, models_dir, manifest)

        for name in ("data", "models"):
            files = manifest[name]
            total = sum(x["bytes"] for x in files)
            print(f"{name}: {len(files)} archivos, {total / 1024**2:.1f} MiB")
        print(f"Carpeta remota nueva: {args.folder}/")
        print(f"Rama remota: {args.branch}")
        print("Datos:", args.data_repo)
        print("Pesos:", args.weights_repo)
        print("Commit local:", git_info(root))

        if args.dry_run:
            print("\nDRY RUN: inventario correcto; no se envio ningun archivo.")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("Instala primero: python -m pip install -U huggingface_hub hf_xet")
            return 2
        api = HfApi()
        try:
            user = api.whoami()
            print("Autenticado en Hugging Face como:", user.get("name", "<desconocido>"))
            for repo, folder in ((args.data_repo, data_dir), (args.weights_repo, models_dir)):
                kind = repo_type(api, repo)
                if args.branch != "main":
                    api.create_branch(repo_id=repo, repo_type=kind,
                                      branch=args.branch, exist_ok=True)
                print(f"\nSubiendo {repo} (tipo {kind}) ...")
                info = api.upload_folder(
                    repo_id=repo,
                    repo_type=kind,
                    folder_path=str(folder),
                    path_in_repo=args.folder,
                    revision=args.branch,
                    commit_message=f"Tesis Grupo25: E0/E1/E2 final SC-test y ST ({args.folder})",
                )
                prefix = "datasets/" if kind == "dataset" else ""
                url = f"https://huggingface.co/{prefix}{repo}/tree/{args.branch}/{args.folder}"
                print("OK:", url)
                print("Commit:", getattr(info, "commit_url", "ver Hub"))
            print("\nSUBIDA COMPLETADA a ambos repositorios.")
        except Exception as exc:
            print(f"\nERROR de Hugging Face ({type(exc).__name__}): {exc}", file=sys.stderr)
            print("Puedes ejecutar otra vez el comando; los archivos locales siguen intactos.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
