"""
hf_hub.py
Gestor centralizado de integración con Hugging Face Hub y control de versiones con DVC.

Permite versionar y sincronizar artefactos pesados (datos y modelos) con los repositorios
oficiales del proyecto en Hugging Face:
  - Dataset Repo: maia-grupo25/sleep-edfx-data
  - Model Repo:   maia-grupo25/sleep-edfx-models

Autenticación:
Lee la variable de entorno HUGGING_FACE_API_KEY (con fallback a HF_TOKEN o HUGGING_FACE_HUB_TOKEN),
soporta carga desde archivo .env y expone tanto funciones programáticas como CLI.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Union

# Intentar cargar variables desde .env si python-dotenv está disponible
try:
    from dotenv import load_dotenv

    _ROOT_DIR = Path(__file__).resolve().parents[2]
    load_dotenv(_ROOT_DIR / ".env")
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [HF-Hub] %(message)s",
)
log = logging.getLogger("hf_hub")

# Repositorios oficiales por defecto
DEFAULT_DATA_REPO = os.getenv("HF_DATA_REPO", "maia-grupo25/sleep-edfx-data")
DEFAULT_MODEL_REPO = os.getenv("HF_MODEL_REPO", "maia-grupo25/sleep-edfx-models")
DEFAULT_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# Mapeo de modelos del proyecto
ROOT_DIR = Path(__file__).resolve().parents[2]
KNOWN_MODELS = {
    "svm": {
        "local_path": ROOT_DIR / "models" / "svm_sleep.joblib",
        "path_in_repo": "models/svm_sleep.joblib",
        "dvc_file": ROOT_DIR / "models" / "svm_sleep.joblib.dvc",
    },
    "lightgbm": {
        "local_path": ROOT_DIR / "models" / "lightgbm_sleep.joblib",
        "path_in_repo": "models/lightgbm_sleep.joblib",
        "dvc_file": ROOT_DIR / "models" / "lightgbm_sleep.joblib.dvc",
    },
    "cnn1d": {
        "local_path": ROOT_DIR / "experiments" / "cnn1d" / "best_cnn1d.pt",
        "path_in_repo": "experiments/cnn1d/best_cnn1d.pt",
        "dvc_file": ROOT_DIR / "experiments" / "cnn1d" / "best_cnn1d.pt.dvc",
    },
}


def get_hf_token(required: bool = False) -> Optional[str]:
    """
    Obtiene el token de autenticación de Hugging Face.
    Prioriza HUGGING_FACE_API_KEY y retrocede a HF_TOKEN / HUGGING_FACE_HUB_TOKEN.
    Inyecta el token en os.environ['HF_TOKEN'] para que el SDK de Hugging Face opere sin fricción.
    """
    token = (
        os.getenv("HUGGING_FACE_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        token = token.strip()
        # Asegurar consistencia para todas las utilidades de Hugging Face
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return token

    if required:
        raise ValueError(
            "No se encontró la variable de entorno 'HUGGING_FACE_API_KEY' (ni 'HF_TOKEN').\n"
            "Configure su token en el archivo .env o en su terminal:\n"
            "  PowerShell: $env:HUGGING_FACE_API_KEY=\"hf_...\"\n"
            "  Bash:       export HUGGING_FACE_API_KEY=\"hf_...\""
        )
    return None


def get_hf_api(token: Optional[str] = None) -> HfApi:
    """Retorna una instancia autenticada de HfApi."""
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "El paquete 'huggingface_hub' no está instalado. Ejecute: pip install huggingface_hub"
        )
    resolved_token = token or get_hf_token(required=True)
    return HfApi(token=resolved_token)


def run_dvc_add(file_or_dir_path: Path) -> bool:
    """
    Ejecuta 'dvc add <path>' sobre un archivo o directorio para generar/actualizar
    el puntero .dvc correspondiente en Git.
    """
    path_obj = Path(file_or_dir_path)
    if not path_obj.exists():
        log.warning("No se puede ejecutar 'dvc add': ruta no existe: %s", path_obj)
        return False

    try:
        cmd = ["dvc", "add", str(path_obj.relative_to(ROOT_DIR))]
    except ValueError:
        cmd = ["dvc", "add", str(path_obj)]

    try:
        log.info("Ejecutando DVC tracking: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            log.info("✓ DVC tracking exitoso para %s", path_obj.name)
            return True
        log.warning("DVC add terminó con advertencia: %s", result.stderr.strip() or result.stdout.strip())
        return False
    except FileNotFoundError:
        log.warning("El comando 'dvc' no está instalado o no se encuentra en el PATH del sistema.")
        return False
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Excepción ejecutando dvc add: %s", exc)
        return False


def verify_hf_connection(
    repo_data: str = DEFAULT_DATA_REPO,
    repo_models: str = DEFAULT_MODEL_REPO,
    repo_type: str = DEFAULT_REPO_TYPE,
) -> Dict[str, bool]:
    """
    Verifica la validez del token y el acceso a los repositorios de Hugging Face.
    """
    token = get_hf_token(required=True)
    api = get_hf_api(token)

    results = {}
    log.info("Verificando autenticación con Hugging Face...")
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname")
        log.info("✓ Autenticado exitosamente como usuario: '%s'", username)
        results["authenticated"] = True
    except Exception as exc:
        log.error("✗ Error autenticando con Hugging Face: %s", exc)
        results["authenticated"] = False
        return results

    for name, r_id in [("data_repo", repo_data), ("models_repo", repo_models)]:
        try:
            repo_info = api.repo_info(repo_id=r_id, repo_type=repo_type)
            log.info("✓ Repositorio '%s' (%s) accesible. SHA: %s", r_id, repo_type, repo_info.sha[:7] if repo_info.sha else "N/A")
            results[name] = True
        except RepositoryNotFoundError:
            log.warning("! Repositorio '%s' no encontrado. Intentando verificar si es posible crearlo...", r_id)
            results[name] = False
        except Exception as exc:
            log.warning("! Error consultando '%s': %s", r_id, exc)
            results[name] = False

    return results


# ── Operaciones sobre Modelos ─────────────────────────────────────────────────

def upload_model_artifact(
    local_path: Union[str, Path],
    repo_id: str = DEFAULT_MODEL_REPO,
    path_in_repo: Optional[str] = None,
    repo_type: str = DEFAULT_REPO_TYPE,
    commit_message: Optional[str] = None,
    update_dvc: bool = True,
) -> str:
    """
    Sube un artefacto de modelo a Hugging Face Hub y opcionalmente actualiza el puntero DVC.
    """
    local_p = Path(local_path).resolve()
    if not local_p.exists():
        raise FileNotFoundError(f"El artefacto del modelo no existe localmente: {local_p}")

    api = get_hf_api()

    # Si no se especifica path_in_repo, preservar la estructura relativa a ROOT_DIR o el nombre
    if not path_in_repo:
        try:
            path_in_repo = str(local_p.relative_to(ROOT_DIR)).replace("\\", "/")
        except ValueError:
            path_in_repo = f"models/{local_p.name}"

    # 1. Versionar en DVC local si se solicita
    if update_dvc:
        run_dvc_add(local_p)

    # 2. Subir binario a Hugging Face Hub
    msg = commit_message or f"Upload model artifact: {path_in_repo}"
    log.info("Subiendo modelo '%s' a HF Hub [%s (%s)]...", local_p.name, repo_id, repo_type)

    commit_info = api.upload_file(
        path_or_fileobj=str(local_p),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
    )
    log.info("✓ Modelo subido exitosamente a: %s", commit_info)

    # 3. Subir puntero .dvc si existe
    dvc_path = local_p.with_suffix(local_p.suffix + ".dvc")
    if dvc_path.exists():
        dvc_in_repo = f"{path_in_repo}.dvc"
        try:
            api.upload_file(
                path_or_fileobj=str(dvc_path),
                path_in_repo=dvc_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Sync DVC pointer for {local_p.name}",
            )
            log.info("✓ Puntero DVC '%s' sincronizado en Hugging Face.", dvc_in_repo)
        except Exception as exc:
            log.warning("No se pudo subir puntero .dvc: %s", exc)

    return str(commit_info)


def download_model_artifact(
    model_key_or_name: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    repo_type: str = DEFAULT_REPO_TYPE,
    destination_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Descarga un modelo desde el repositorio oficial de Hugging Face.
    Soporta alias: 'svm', 'lightgbm', 'cnn1d' o el nombre exacto del archivo.
    """
    key = model_key_or_name.lower().strip()
    if key in KNOWN_MODELS:
        info = KNOWN_MODELS[key]
        target_path = info["local_path"]
        filename_in_repo = info["path_in_repo"]
    else:
        filename_in_repo = model_key_or_name
        dest_folder = destination_dir or (ROOT_DIR / "models")
        target_path = dest_folder / Path(model_key_or_name).name

    if target_path.exists() and not force:
        log.info("Modelo ya existe localmente en: %s", target_path)
        return target_path

    token = get_hf_token(required=False)
    log.info("Descargando '%s' desde Hugging Face Hub [%s]...", filename_in_repo, repo_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename_in_repo,
        repo_type=repo_type,
        token=token,
        local_dir=str(ROOT_DIR),
        local_dir_use_symlinks=False,
    )
    log.info("✓ Descarga completada: %s", downloaded)
    return Path(downloaded)


def sync_all_models_to_hf(
    repo_id: str = DEFAULT_MODEL_REPO,
    repo_type: str = DEFAULT_REPO_TYPE,
    update_dvc: bool = True,
) -> Dict[str, str]:
    """
    Escanea los modelos locales existentes y los sincroniza con Hugging Face Hub.
    """
    results = {}
    for key, info in KNOWN_MODELS.items():
        local_p = info["local_path"]
        if local_p.exists():
            log.info("Sincronizando modelo local '%s' (%s)...", key, local_p.name)
            res = upload_model_artifact(
                local_path=local_p,
                repo_id=repo_id,
                path_in_repo=info["path_in_repo"],
                repo_type=repo_type,
                commit_message=f"Sync model {key} artifact ({local_p.name})",
                update_dvc=update_dvc,
            )
            results[key] = res
        else:
            log.warning("Modelo '%s' no encontrado localmente en: %s", key, local_p)
    return results


def pull_all_models_from_hf(
    repo_id: str = DEFAULT_MODEL_REPO,
    repo_type: str = DEFAULT_REPO_TYPE,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Descarga todos los modelos del proyecto desde Hugging Face Hub si faltan en disco.
    """
    downloaded = {}
    for key in KNOWN_MODELS:
        try:
            path = download_model_artifact(
                model_key_or_name=key,
                repo_id=repo_id,
                repo_type=repo_type,
                force=force,
            )
            downloaded[key] = path
        except Exception as exc:
            log.warning("No se pudo descargar modelo '%s': %s", key, exc)
    return downloaded


# ── Operaciones sobre Datasets ────────────────────────────────────────────────

def upload_dataset_artifacts(
    source_path: Union[str, Path],
    repo_id: str = DEFAULT_DATA_REPO,
    path_in_repo: Optional[str] = None,
    repo_type: str = DEFAULT_REPO_TYPE,
    commit_message: Optional[str] = None,
    update_dvc: bool = True,
) -> str:
    """
    Sube un archivo o directorio de datos a Hugging Face Hub y opcionalmente
    lo versiona con DVC.
    """
    src_p = Path(source_path).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"Ruta de dataset no encontrada: {src_p}")

    api = get_hf_api()

    if update_dvc:
        run_dvc_add(src_p)

    msg = commit_message or f"Upload dataset artifacts from {src_p.name}"

    if src_p.is_file():
        dest = path_in_repo or src_p.name
        log.info("Subiendo archivo de datos '%s' a HF Hub [%s]...", src_p.name, repo_id)
        info = api.upload_file(
            path_or_fileobj=str(src_p),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=msg,
        )
    else:
        dest = path_in_repo or ""
        log.info("Subiendo carpeta de datos '%s' a HF Hub [%s/%s]...", src_p.name, repo_id, dest)
        info = api.upload_folder(
            folder_path=str(src_p),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=msg,
        )

    log.info("✓ Dataset subido exitosamente: %s", info)
    return str(info)


def download_dataset_artifacts(
    repo_id: str = DEFAULT_DATA_REPO,
    repo_type: str = DEFAULT_REPO_TYPE,
    local_dir: Optional[Union[str, Path]] = None,
    subfolder: Optional[str] = None,
) -> Path:
    """
    Descarga el dataset completo o un subdirectorio desde Hugging Face Hub.
    """
    dest = Path(local_dir) if local_dir else ROOT_DIR / "data"
    dest.mkdir(parents=True, exist_ok=True)
    token = get_hf_token(required=False)

    log.info("Descargando dataset desde HF Hub [%s] hacia '%s'...", repo_id, dest)
    res = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    log.info("✓ Dataset descargado en: %s", res)
    return Path(res)


# ── CLI Runner ────────────────────────────────────────────────────────────────

def main():
    """Interfaz de línea de comandos para gestión de Hugging Face y DVC."""
    parser = argparse.ArgumentParser(
        description="Gestor de sincronización de datos y modelos con Hugging Face Hub & DVC."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: verify
    sub_verify = subparsers.add_parser("verify", help="Verifica credenciales y conexión a los repositorios.")
    sub_verify.add_argument("--repo-data", default=DEFAULT_DATA_REPO)
    sub_verify.add_argument("--repo-models", default=DEFAULT_MODEL_REPO)

    # Subcomando: push-models
    sub_push_m = subparsers.add_parser("push-models", help="Sube todos los modelos locales a Hugging Face.")
    sub_push_m.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    sub_push_m.add_argument("--no-dvc", action="store_true", help="Omitir dvc add antes de subir")

    # Subcomando: pull-models
    sub_pull_m = subparsers.add_parser("pull-models", help="Descarga los modelos del proyecto desde Hugging Face.")
    sub_pull_m.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    sub_pull_m.add_argument("--force", action="store_true", help="Sobrescribir si ya existen localmente")

    # Subcomando: push-data
    sub_push_d = subparsers.add_parser("push-data", help="Sube una carpeta o archivo de datos a Hugging Face.")
    sub_push_d.add_argument("--path", required=True, help="Ruta local del archivo o carpeta a subir")
    sub_push_d.add_argument("--repo-id", default=DEFAULT_DATA_REPO)
    sub_push_d.add_argument("--path-in-repo", default=None, help="Ruta destino dentro del repositorio")
    sub_push_d.add_argument("--no-dvc", action="store_true", help="Omitir dvc add")

    # Subcomando: pull-data
    sub_pull_d = subparsers.add_parser("pull-data", help="Descarga dataset completo desde Hugging Face.")
    sub_pull_d.add_argument("--repo-id", default=DEFAULT_DATA_REPO)
    sub_pull_d.add_argument("--local-dir", default=str(ROOT_DIR / "data"), help="Directorio destino local")

    args = parser.parse_args()

    if args.command == "verify":
        verify_hf_connection(args.repo_data, args.repo_models)
    elif args.command == "push-models":
        sync_all_models_to_hf(repo_id=args.repo_id, update_dvc=not args.no_dvc)
    elif args.command == "pull-models":
        pull_all_models_from_hf(repo_id=args.repo_id, force=args.force)
    elif args.command == "push-data":
        upload_dataset_artifacts(
            source_path=args.path,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            update_dvc=not args.no_dvc,
        )
    elif args.command == "pull-data":
        download_dataset_artifacts(repo_id=args.repo_id, local_dir=args.local_dir)


if __name__ == "__main__":
    main()
