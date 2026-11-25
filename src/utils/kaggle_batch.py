"""
kaggle_batch.py
Gestor centralizado de integración con Kaggle Batch API (Kernels API) para:
  1. Consultar estado de notebooks remotos ya ejecutados en Kaggle (E0, E1, E2).
  2. Descargar predicciones, métricas y artefactos precalculados (`kernels_output`).
  3. Disparar ejecuciones batch remotas en GPU de Kaggle (`kernels_push`).

Autenticación:
Lee las variables de entorno KAGGLE_USERNAME y KAGGLE_KEY desde el archivo `.env`
o desde el entorno del sistema / Docker Compose.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_KAGGLE_LOCK = threading.Lock()


def _call_kaggle_with_retry(fn: Callable[[], Any], retries: int = 3) -> Any:
    """Ejecuta llamadas al SDK de Kaggle de forma thread-safe con reintento ante errores SSL/EOF transitorios."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with _KAGGLE_LOCK:
                return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            transient = any(k in msg for k in ("SSLEOFError", "UNEXPECTED_EOF", "ConnectionReset", "RemoteDisconnected", "Max retries exceeded"))
            if not transient or attempt == retries:
                raise
            wait_s = 0.6 * attempt
            log.warning("Error SSL/red transitorio en Kaggle API (intento %d/%d). Reintentando en %.1fs...", attempt, retries, wait_s)
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc

ROOT_DIR = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [Kaggle-Batch] %(message)s",
)
log = logging.getLogger("kaggle_batch")

# Directorio local donde se cachean los outputs descargados de Kaggle
KAGGLE_OUTPUTS_DIR = ROOT_DIR / "outputs" / "kaggle_batch"

# Slugs recomendados por defecto (el usuario reemplaza KAGGLE_USERNAME en .env)
DEFAULT_SLUGS = {
    "e0": "sleep-edfx-e0-batch",
    "e1": "sleep-edfx-e1-batch",
    "e2": "sleep-edfx-e2-batch",
}

ENV_SLUG_KEYS = {
    "e0": "KAGGLE_KERNEL_E0",
    "e1": "KAGGLE_KERNEL_E1",
    "e2": "KAGGLE_KERNEL_E2",
}


def _clean_env(key: str) -> str:
    val = os.getenv(key, "").strip()
    if val.lower().startswith(("tu_", "your_", "<")):
        return ""
    return val


def get_kaggle_credentials() -> Dict[str, str]:
    """
    Obtiene credenciales de Kaggle desde .env / entorno.
    Soporta tanto el nuevo token Bearer (`KAGGLE_API_TOKEN` o `KAGGLE_KEY=KGAT_...`)
    como el esquema legacy (`KAGGLE_USERNAME` + `KAGGLE_KEY` hex de 32 caracteres).
    """
    username = _clean_env("KAGGLE_USERNAME")
    api_token = _clean_env("KAGGLE_API_TOKEN")
    key = _clean_env("KAGGLE_KEY")

    # Si el usuario pegó un token nuevo KGAT_... en KAGGLE_KEY, mapearlo a KAGGLE_API_TOKEN
    if not api_token and key.startswith("KGAT_"):
        api_token = key

    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if api_token:
        os.environ["KAGGLE_API_TOKEN"] = api_token
        os.environ.pop("KAGGLE_KEY", None)
    elif key:
        os.environ["KAGGLE_KEY"] = key

    return {
        "username": username,
        "key": api_token or key,
        "auth_mode": "api_token" if api_token else ("legacy_key" if key else "none"),
    }


def resolve_kernel_slug(model_id_or_slug: str) -> str:
    """
    Resuelve un alias ('e0', 'e1', 'e2') a su slug completo de Kaggle ('usuario/slug'),
    o retorna el slug directamente si ya contiene '/'.
    """
    raw = model_id_or_slug.strip()
    if "/" in raw:
        return raw

    key = raw.lower()
    if key not in DEFAULT_SLUGS:
        raise ValueError(
            f"Modelo Kaggle '{model_id_or_slug}' no reconocido. "
            f"Opciones válidas: {list(DEFAULT_SLUGS.keys())} o un slug 'usuario/notebook'."
        )

    env_slug = _clean_env(ENV_SLUG_KEYS[key])
    if env_slug:
        if "/" in env_slug:
            return env_slug
        username = get_kaggle_credentials()["username"]
        return f"{username}/{env_slug}" if username else env_slug

    username = get_kaggle_credentials()["username"]
    if not username:
        raise ValueError(
            f"Configure KAGGLE_USERNAME o {ENV_SLUG_KEYS[key]} en el archivo .env "
            f"para resolver el slug del modelo '{key}'."
        )
    return f"{username}/{DEFAULT_SLUGS[key]}"


def get_kaggle_config() -> Dict[str, Any]:
    """Retorna el estado de configuración de Kaggle sin exponer la llave secreta."""
    creds = get_kaggle_credentials()
    configured = bool(creds["username"] and creds["key"])
    kernels: Dict[str, Dict[str, Any]] = {}
    for model_id, default_name in DEFAULT_SLUGS.items():
        env_val = _clean_env(ENV_SLUG_KEYS[model_id])
        if env_val and "/" in env_val:
            slug = env_val
        elif creds["username"]:
            slug = f"{creds['username']}/{env_val or default_name}"
        else:
            slug = f"<KAGGLE_USERNAME>/{env_val or default_name}"
        kernels[model_id] = {
            "model_id": model_id,
            "env_var": ENV_SLUG_KEYS[model_id],
            "slug": slug,
            "kernel_dir": str(ROOT_DIR / "kaggle" / model_id),
        }

    return {
        "configured": configured,
        "username": creds["username"] or None,
        "has_key": bool(creds["key"]),
        "kernels": kernels,
    }


def get_kaggle_api():
    """Retorna una instancia autenticada de KaggleApi."""
    creds = get_kaggle_credentials()
    if not creds["username"] or not creds["key"]:
        raise RuntimeError(
            "Credenciales de Kaggle no configuradas. "
            "Diligencia KAGGLE_USERNAME y KAGGLE_KEY en el archivo .env."
        )

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError(
            "El paquete 'kaggle' no está instalado. Ejecuta: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def get_kernel_status(model_id_or_slug: str) -> Dict[str, Any]:
    """Consulta el estado actual de un kernel/notebook en Kaggle."""
    slug = resolve_kernel_slug(model_id_or_slug)
    log.info("Consultando estado de kernel Kaggle: %s", slug)

    resp = _call_kaggle_with_retry(lambda: get_kaggle_api().kernels_status(slug))
    # Compatibilidad entre versiones del SDK de Kaggle (objeto o dict)
    if isinstance(resp, dict):
        status_str = str(resp.get("status", "unknown"))
        failure_msg = resp.get("failureMessage") or ""
    else:
        status_str = str(getattr(resp, "status", resp))
        failure_msg = str(getattr(resp, "failureMessage", "") or "")

    return {
        "slug": slug,
        "status": status_str,
        "failure_message": failure_msg or None,
        "kaggle_url": f"https://www.kaggle.com/code/{slug}",
    }


def fetch_kernel_outputs(
    model_id_or_slug: str,
    dest_dir: Optional[Path] = None,
    force: bool = True,
) -> Dict[str, Any]:
    """
    Descarga los archivos de salida (predicciones, JSON, CSV, métricas)
    ya generados por el notebook entrenado en Kaggle.
    """
    slug = resolve_kernel_slug(model_id_or_slug)

    safe_name = slug.replace("/", "__")
    out_dir = Path(dest_dir) if dest_dir else (KAGGLE_OUTPUTS_DIR / safe_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Descargando outputs del kernel Kaggle '%s' en '%s'...", slug, out_dir)
    _call_kaggle_with_retry(lambda: get_kaggle_api().kernels_output(slug, path=str(out_dir), force=force, quiet=False))

    files: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out_dir).as_posix()
        files.append({"file": rel, "size_bytes": p.stat().st_size})
        if p.suffix.lower() == ".json" and p.stat().st_size <= 512_000:
            try:
                summaries[rel] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass

    return {
        "slug": slug,
        "output_dir": str(out_dir),
        "file_count": len(files),
        "files": files,
        "json_summaries": summaries,
        "kaggle_url": f"https://www.kaggle.com/code/{slug}",
    }


def trigger_kaggle_kernel(
    model_id: str,
    kernel_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Actualiza el ID del usuario en kernel-metadata.json y dispara la ejecución
    remota batch del notebook en Kaggle mediante `kernels_push`.
    """
    key = model_id.lower().strip()
    slug = resolve_kernel_slug(key)

    k_dir = Path(kernel_dir) if kernel_dir else (ROOT_DIR / "kaggle" / key)
    meta_file = k_dir / "kernel-metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(
            f"No se encontró '{meta_file}'. Verifica la carpeta de plantilla kaggle/{key}/."
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["id"] = slug
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("Enviando ejecución batch remota a Kaggle para '%s'...", slug)
    resp = _call_kaggle_with_retry(lambda: get_kaggle_api().kernels_push(str(k_dir)))
    ref = getattr(resp, "ref", None) or (resp.get("ref") if isinstance(resp, dict) else slug)
    url = getattr(resp, "url", None) or (
        resp.get("url") if isinstance(resp, dict) else f"https://www.kaggle.com/code/{slug}"
    )

    return {
        "slug": slug,
        "ref": str(ref),
        "status": "queued",
        "kaggle_url": str(url),
        "message": f"Ejecución remota disparada en Kaggle para {slug}.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="CLI de integración Kaggle Batch API para modelos E0, E1 y E2."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="Muestra configuración actual de Kaggle y slugs.")

    p_status = sub.add_parser("status", help="Consulta estado de uno o todos los kernels en Kaggle.")
    p_status.add_argument("--model", required=True, help="e0, e1, e2, all o usuario/slug")

    p_out = sub.add_parser("outputs", help="Descarga predicciones/outputs de uno o todos los kernels.")
    p_out.add_argument("--model", required=True, help="e0, e1, e2, all o usuario/slug")

    p_push = sub.add_parser("push", help="Dispara ejecución batch remota en Kaggle.")
    p_push.add_argument("--model", required=True, choices=["e0", "e1", "e2", "all"])

    args = parser.parse_args()
    if args.command == "config":
        print(json.dumps(get_kaggle_config(), indent=2, ensure_ascii=False))
    elif args.command == "status":
        targets = list(DEFAULT_SLUGS.keys()) if args.model.lower() == "all" else [args.model]
        res = {m: get_kernel_status(m) for m in targets} if len(targets) > 1 else get_kernel_status(targets[0])
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.command == "outputs":
        targets = list(DEFAULT_SLUGS.keys()) if args.model.lower() == "all" else [args.model]
        res = {m: fetch_kernel_outputs(m) for m in targets} if len(targets) > 1 else fetch_kernel_outputs(targets[0])
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.command == "push":
        targets = list(DEFAULT_SLUGS.keys()) if args.model.lower() == "all" else [args.model]
        res = {m: trigger_kaggle_kernel(m) for m in targets} if len(targets) > 1 else trigger_kaggle_kernel(targets[0])
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
