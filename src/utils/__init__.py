"""
Módulo de utilidades generales del proyecto.
"""


def __getattr__(name: str):
    if name in (
        "get_hf_token",
        "get_hf_api",
        "upload_model_artifact",
        "download_model_artifact",
        "upload_dataset_artifacts",
        "download_dataset_artifacts",
        "sync_all_models_to_hf",
        "pull_all_models_from_hf",
        "verify_hf_connection",
    ):
        from src.utils import hf_hub

        return getattr(hf_hub, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
