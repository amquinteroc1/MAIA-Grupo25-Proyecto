"""
predict.py
Módulo de inferencia para clasificación de etapas de sueño en ventanas de 30s.
Soporta los modelos de tesis (E0, E1, E2 con normalización SC-train y contexto
temporal de 5 épocas) y los modelos previos del microproyecto (SVM, LightGBM, CNN 1D).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.data.features import extraer_features

# ── Configuración y Rutas ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

CANALES_ML = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
CLASES_CANONICAS = ["Wake", "N1", "N2", "N3", "REM"]

DESCRIPCION_ETAPAS = {
    "Wake": "Vigilia",
    "N1": "Fase N1 (Ligero)",
    "N2": "Fase N2 (Intermedio)",
    "N3": "Fase N3 (Profundo)",
    "REM": "Fase REM (Sueño paradójico)",
}

NORM_PARAMS_PATH = ROOT / "data" / "processed" / "meta" / "norm_params.npz"

THESIS_ARCHS = ("e0", "e1", "e2")
THESIS_SEEDS = ("42", "123", "2026")

MODEL_PATHS: Dict[str, Path] = {
    "e0": ROOT / "experiments" / "E0" / "seed42" / "best_model_E0_seed42.pt",
    "e1": ROOT / "experiments" / "E1" / "seed42" / "best_model_E1_seed42.pt",
    "e2": ROOT / "experiments" / "E2" / "seed42" / "best_model_E2_seed42.pt",
    **{
        f"{arch}_seed{seed}": ROOT / "experiments" / arch.upper() / f"seed{seed}" / f"best_model_{arch.upper()}_seed{seed}.pt"
        for arch in THESIS_ARCHS
        for seed in THESIS_SEEDS
    },
    "svm": ROOT / "models" / "svm_sleep.joblib",
    "lightgbm": ROOT / "models" / "lightgbm_sleep.joblib",
    "cnn1d": ROOT / "experiments" / "cnn1d" / "best_cnn1d.pt",
}

THESIS_MODELS = tuple(
    list(THESIS_ARCHS) + [f"{arch}_seed{seed}" for arch in THESIS_ARCHS for seed in THESIS_SEEDS]
)

# Cache en memoria de modelos y parámetros de normalización
_MODEL_CACHE: Dict[str, Any] = {}
_NORM_CACHE: Optional[Tuple[np.ndarray, np.ndarray]] = None


# ── Normalización oficial de tesis (SC-train) ─────────────────────────────────
def cargar_norm_params() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Carga (media, std) por canal estimados en SC-train para E0/E1/E2."""
    global _NORM_CACHE
    if _NORM_CACHE is not None:
        return _NORM_CACHE

    ruta = NORM_PARAMS_PATH
    if not ruta.exists():
        try:
            from src.utils.hf_hub import download_norm_params

            ruta = download_norm_params()
        except Exception as exc:
            print(f"[Hugging Face] No se pudo descargar norm_params.npz: {exc}")
            return None

    if ruta.exists():
        params = np.load(ruta)
        media = params["media"].astype(np.float32)
        std = params["std"].astype(np.float32)
        _NORM_CACHE = (media, std)
        return _NORM_CACHE
    return None


# ── Catálogo de Modelos ───────────────────────────────────────────────────────
def get_model_catalog() -> Dict[str, Dict[str, Any]]:
    """Retorna información y disponibilidad de todos los modelos soportados."""
    catalog: Dict[str, Dict[str, Any]] = {}

    # E0 - Tesis (CNN 1D sin contexto)
    e0_path = MODEL_PATHS["e0"]
    catalog["e0"] = {
        "id": "e0",
        "name": "E0 — CNN 1D Época Individual (Tesis)",
        "type": "Deep Learning (Tesis - Baseline)",
        "path": str(e0_path),
        "available": e0_path.exists() and TORCH_AVAILABLE,
        "classes": CLASES_CANONICAS,
        "input_type": "1 época (3×3000 @ 100 Hz, Z-score SC-train)",
        "notes": None if TORCH_AVAILABLE else "Requiere paquete PyTorch (torch)",
    }

    # E1 - Tesis (CNN 1D + Concatenación 5 épocas)
    e1_path = MODEL_PATHS["e1"]
    catalog["e1"] = {
        "id": "e1",
        "name": "E1 — CNN 1D + Concatenación 5 Épocas (Tesis)",
        "type": "Deep Learning (Tesis - Contexto Concat)",
        "path": str(e1_path),
        "available": e1_path.exists() and TORCH_AVAILABLE,
        "classes": CLASES_CANONICAS,
        "input_type": "5 épocas consecutivas (5×3×3000 @ 100 Hz)",
        "notes": None if TORCH_AVAILABLE else "Requiere paquete PyTorch (torch)",
    }

    # E2 - Tesis (CNN 1D + GRU 5 épocas)
    e2_path = MODEL_PATHS["e2"]
    catalog["e2"] = {
        "id": "e2",
        "name": "E2 — CNN 1D + GRU 5 Épocas (Tesis - Mejor Modelo)",
        "type": "Deep Learning (Tesis - Contexto GRU)",
        "path": str(e2_path),
        "available": e2_path.exists() and TORCH_AVAILABLE,
        "classes": CLASES_CANONICAS,
        "input_type": "5 épocas consecutivas (5×3×3000 @ 100 Hz)",
        "notes": None if TORCH_AVAILABLE else "Requiere paquete PyTorch (torch)",
    }

    # 9 corridas oficiales de tesis (E0, E1, E2 × semillas 42, 123, 2026) — emparejadas con MLflow
    _arch_meta = {
        "e0": ("E0 — CNN 1D Base", "Deep Learning (Tesis - Baseline)", "1 época (3×3000 @ 100 Hz, Z-score SC-train)"),
        "e1": ("E1 — CNN 1D + Concat", "Deep Learning (Tesis - Contexto Concat)", "5 épocas consecutivas (5×3×3000 @ 100 Hz)"),
        "e2": ("E2 — CNN 1D + GRU", "Deep Learning (Tesis - Contexto GRU)", "5 épocas consecutivas (5×3×3000 @ 100 Hz)"),
    }
    for arch in THESIS_ARCHS:
        label, mtype, inp = _arch_meta[arch]
        for seed in THESIS_SEEDS:
            mid = f"{arch}_seed{seed}"
            mpath = MODEL_PATHS[mid]
            catalog[mid] = {
                "id": mid,
                "name": f"{label} (Seed {seed})",
                "type": mtype,
                "path": str(mpath),
                "available": mpath.exists() and TORCH_AVAILABLE,
                "classes": CLASES_CANONICAS,
                "input_type": inp,
                "notes": None if TORCH_AVAILABLE else "Requiere paquete PyTorch (torch)",
            }

    # SVM
    svm_path = MODEL_PATHS["svm"]
    catalog["svm"] = {
        "id": "svm",
        "name": "SVM (Support Vector Machine)",
        "type": "Machine Learning (Baseline previo)",
        "path": str(svm_path),
        "available": svm_path.exists(),
        "classes": CLASES_CANONICAS,
        "input_type": "30 Features (Espectrales / Temporales)",
        "notes": None,
    }

    # LightGBM
    lgb_path = MODEL_PATHS["lightgbm"]
    catalog["lightgbm"] = {
        "id": "lightgbm",
        "name": "LightGBM (Gradient Boosting)",
        "type": "Machine Learning (Ensemble previo)",
        "path": str(lgb_path),
        "available": lgb_path.exists(),
        "classes": CLASES_CANONICAS,
        "input_type": "30 Features (Espectrales / Temporales)",
        "notes": None,
    }

    # CNN 1D previo
    cnn_path = MODEL_PATHS["cnn1d"]
    catalog["cnn1d"] = {
        "id": "cnn1d",
        "name": "CNN 1D (Microproyecto previo)",
        "type": "Deep Learning (End-to-End)",
        "path": str(cnn_path),
        "available": cnn_path.exists() and TORCH_AVAILABLE,
        "classes": CLASES_CANONICAS,
        "input_type": "Señal cruda 3 canales x 3000 muestras (100 Hz)",
        "notes": None if TORCH_AVAILABLE else "Requiere paquete PyTorch (torch)",
    }

    return catalog


# ── Carga de Modelos con Cache ────────────────────────────────────────────────
def cargar_modelo(ruta: Path):
    """Carga un modelo serializado con joblib (SVM o LightGBM)."""
    return joblib.load(Path(ruta))


def cargar_modelo_tesis(model_key: str, ruta: Path):
    """Carga los checkpoints oficiales de tesis E0, E1 o E2 (cualquier semilla) en modo evaluación."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    base_arch = model_key.lower().strip().split("_")[0]
    if base_arch == "e0":
        from src.models.cnn1d import SleepCNN1D

        model = SleepCNN1D()
    elif base_arch == "e1":
        from src.models.cnn1d_e1 import SleepCNN1D_E1

        model = SleepCNN1D_E1()
    elif base_arch == "e2":
        from src.models.cnn1d_e2 import SleepCNN1D_E2

        model = SleepCNN1D_E2()
    else:
        raise ValueError(f"Modelo de tesis no soportado: {model_key}")

    ckpt = torch.load(Path(ruta), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def obtener_modelo(model_key: str):
    """Obtiene el modelo desde cache o lo carga desde disco (con autodescarga HF)."""
    key = model_key.lower().strip()
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    catalog = get_model_catalog()
    if key not in catalog:
        raise ValueError(
            f"Modelo '{model_key}' no reconocido. Opciones válidas: {list(catalog.keys())}"
        )

    info = catalog[key]
    ruta = Path(info["path"])

    # Si el modelo no existe localmente, intentar descargarlo bajo demanda desde Hugging Face Hub
    if not ruta.exists():
        try:
            from src.utils.hf_hub import download_model_artifact

            print(f"[Hugging Face] Modelo '{key}' no encontrado localmente. Descargando desde repo oficial...")
            download_model_artifact(key)
            catalog = get_model_catalog()
            info = catalog[key]
        except Exception as exc:
            print(f"[Hugging Face] No fue posible descargar el modelo '{key}' automáticamente: {exc}")

    if not info["available"]:
        msg = f"El modelo '{key}' no está disponible actualmente."
        if info.get("notes"):
            msg += f" ({info['notes']})"
        raise RuntimeError(msg)

    if key in ("svm", "lightgbm"):
        model = cargar_modelo(ruta)
    elif key == "cnn1d":
        model = cargar_modelo_cnn(ruta)
    elif key in THESIS_MODELS:
        model = cargar_modelo_tesis(key, ruta)
    else:
        raise ValueError(f"Tipo de modelo no manejado: {key}")

    _MODEL_CACHE[key] = model
    return model


# ── Inferencia para Modelos Tabulares (SVM y LightGBM) ───────────────────────
def preparar_features(raw, inicio: float, duracion: float = 30.0) -> pd.DataFrame:
    """Extrae las 30 características espectrales/temporales para modelos tabulares."""
    fs = raw.info["sfreq"]
    start = int(inicio * fs)
    stop = int((inicio + duracion) * fs)

    fila = {}
    for canal in CANALES_ML:
        if canal not in raw.ch_names:
            raise ValueError(f"Canal no encontrado en EDF: {canal}")

        signal = raw.get_data(picks=[canal], start=start, stop=stop)[0]
        features = extraer_features(signal, fs)
        nombre_canal = canal.replace(" ", "_").replace("-", "_")

        for nombre_feature, valor in features.items():
            fila[f"{nombre_canal}_{nombre_feature}"] = valor

    return pd.DataFrame([fila])


def predecir(modelo, X: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Predice la etapa de sueño y probabilidades usando SVM o LightGBM."""
    probabilidades = modelo.predict_proba(X)[0]
    clases = list(modelo.classes_)

    i_max = int(probabilidades.argmax())
    etapa = str(clases[i_max])

    probs = {str(clase): float(prob) for clase, prob in zip(clases, probabilidades)}

    for c in CLASES_CANONICAS:
        if c not in probs:
            probs[c] = 0.0

    return etapa, probs


# ── Inferencia CNN 1D / Tesis (E0, E1, E2) ────────────────────────────────────
def cargar_modelo_cnn(ruta: Path):
    """Carga los pesos del modelo PyTorch CNN 1D en modo evaluación."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    from src.models.cnn1d import build_model

    ckpt = torch.load(Path(ruta), map_location="cpu", weights_only=False)
    dropout = ckpt.get("args", {}).get("dropout", 0.5) if isinstance(ckpt, dict) else 0.5
    model = build_model(n_channels=3, n_classes=5, dropout=dropout)

    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _extraer_epoca_cruda(
    raw, inicio: float, duracion: float = 30.0, n_muestras: int = 3000, use_sc_train_norm: bool = False
) -> np.ndarray:
    """Extrae una época (3, 3000) y aplica normalización Z-score (SC-train o local)."""
    fs = float(raw.info["sfreq"])
    total_samples = raw.n_times
    start = max(0, int(round(inicio * fs)))
    stop = min(total_samples, int(round((inicio + duracion) * fs)))
    if stop <= start:
        stop = min(total_samples, start + int(round(duracion * fs)))

    for canal in CANALES_ML:
        if canal not in raw.ch_names:
            raise ValueError(f"Canal no encontrado en EDF: {canal}")

    data = raw.get_data(picks=CANALES_ML, start=start, stop=stop).astype(np.float32)

    if data.shape[1] != n_muestras:
        from scipy.signal import resample

        data = resample(data, n_muestras, axis=1).astype(np.float32)

    norm = cargar_norm_params() if use_sc_train_norm else None
    if norm is not None:
        media, std = norm
        for ch in range(data.shape[0]):
            if std[ch] > 0:
                data[ch] = (data[ch] - media[ch]) / std[ch]
    else:
        for ch in range(data.shape[0]):
            mu = float(data[ch].mean())
            sigma = float(data[ch].std())
            if sigma > 0:
                data[ch] = (data[ch] - mu) / sigma

    return data


def preparar_tensor_cnn(
    raw, inicio: float, duracion: float = 30.0, n_muestras: int = 3000, use_sc_train_norm: bool = False
):
    """Retorna tensor (1, 3, 3000) para CNN 1D / E0."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")
    data = _extraer_epoca_cruda(
        raw, inicio=inicio, duracion=duracion, n_muestras=n_muestras, use_sc_train_norm=use_sc_train_norm
    )
    return torch.from_numpy(data).unsqueeze(0)


def preparar_tensor_secuencia(
    raw, inicio: float, duracion: float = 30.0, n_muestras: int = 3000, seq_len: int = 5
):
    """
    Construye la ventana contextual de 5 épocas [t-2, t-1, t, t+1, t+2]
    con normalización SC-train y retorna tensor (1, 5, 3, 3000) para E1 y E2.
    En los bordes del archivo repite la primera/última época disponible.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    dur_total = float(raw.times[-1]) if len(raw.times) > 0 else duracion
    max_start = max(0.0, dur_total - duracion)
    half = seq_len // 2

    epocas = []
    for offset in range(-half, half + 1):
        t_start = min(max(0.0, inicio + offset * duracion), max_start)
        ep = _extraer_epoca_cruda(
            raw, inicio=t_start, duracion=duracion, n_muestras=n_muestras, use_sc_train_norm=True
        )
        epocas.append(ep)

    seq_arr = np.stack(epocas, axis=0)  # (5, 3, 3000)
    return torch.from_numpy(seq_arr).unsqueeze(0)  # (1, 5, 3, 3000)


def predecir_cnn(modelo, x_tensor) -> Tuple[str, Dict[str, float]]:
    """Predice la etapa de sueño y distribución de probabilidad usando PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    with torch.no_grad():
        logits = modelo(x_tensor)
        probabilidades = torch.softmax(logits, dim=1)[0].cpu().numpy()

    i_max = int(probabilidades.argmax())
    etapa = CLASES_CANONICAS[i_max]

    probs = {clase: float(prob) for clase, prob in zip(CLASES_CANONICAS, probabilidades)}
    return etapa, probs


# ── Función Unificada de Inferencia ───────────────────────────────────────────
def ejecutar_inferencia_modelo(
    model_key: str,
    raw,
    inicio: float,
    duracion: float = 30.0,
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Ejecuta la inferencia para cualquier modelo soportado (e0, e1, e2, svm, lightgbm, cnn1d).
    """
    key = model_key.lower().strip()
    modelo = obtener_modelo(key)

    base_arch = key.split("_")[0]
    if key in ("svm", "lightgbm"):
        if features_df is None:
            features_df = preparar_features(raw, inicio=inicio, duracion=duracion)
        etapa, probs = predecir(modelo, features_df)
    elif key == "cnn1d":
        x_tensor = preparar_tensor_cnn(raw, inicio=inicio, duracion=duracion, use_sc_train_norm=False)
        etapa, probs = predecir_cnn(modelo, x_tensor)
    elif base_arch == "e0":
        x_tensor = preparar_tensor_cnn(raw, inicio=inicio, duracion=duracion, use_sc_train_norm=True)
        etapa, probs = predecir_cnn(modelo, x_tensor)
    elif base_arch in ("e1", "e2"):
        x_tensor = preparar_tensor_secuencia(raw, inicio=inicio, duracion=duracion, seq_len=5)
        etapa, probs = predecir_cnn(modelo, x_tensor)
    else:
        raise ValueError(f"Modelo desconocido: {key}")

    confianza = float(probs.get(etapa, 0.0))
    descripcion = DESCRIPCION_ETAPAS.get(etapa, "")

    return {
        "model_id": key,
        "stage": etapa,
        "stage_description": descripcion,
        "confidence": round(confianza, 4),
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
    }
