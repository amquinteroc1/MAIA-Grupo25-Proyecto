"""
predict.py
Módulo de inferencia para clasificación de etapas de sueño en ventanas de 30s.
Soporta modelos clásicos (SVM, LightGBM) y Deep Learning (CNN 1D).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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

MODEL_PATHS = {
    "svm": ROOT / "models" / "svm_sleep.joblib",
    "lightgbm": ROOT / "models" / "lightgbm_sleep.joblib",
    "cnn1d": ROOT / "experiments" / "cnn1d" / "best_cnn1d.pt",
}

# Cache en memoria de modelos cargados
_MODEL_CACHE: Dict[str, Any] = {}


# ── Catálogo de Modelos ───────────────────────────────────────────────────────
def get_model_catalog() -> Dict[str, Dict[str, Any]]:
    """Retorna información y disponibilidad de todos los modelos soportados."""
    catalog = {}

    # SVM
    svm_path = MODEL_PATHS["svm"]
    catalog["svm"] = {
        "id": "svm",
        "name": "SVM (Support Vector Machine)",
        "type": "Machine Learning (Baseline)",
        "path": str(svm_path),
        "available": svm_path.exists(),
        "classes": CLASES_CANONICAS,
        "input_type": "30 Features (Espectrales / Temporales)",
    }

    # LightGBM
    lgb_path = MODEL_PATHS["lightgbm"]
    catalog["lightgbm"] = {
        "id": "lightgbm",
        "name": "LightGBM (Gradient Boosting)",
        "type": "Machine Learning (Ensemble)",
        "path": str(lgb_path),
        "available": lgb_path.exists(),
        "classes": CLASES_CANONICAS,
        "input_type": "30 Features (Espectrales / Temporales)",
    }

    # CNN 1D
    cnn_path = MODEL_PATHS["cnn1d"]
    catalog["cnn1d"] = {
        "id": "cnn1d",
        "name": "CNN 1D (Convolutional Neural Network)",
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


def obtener_modelo(model_key: str):
    """Obtiene el modelo desde cache o lo carga desde disco."""
    key = model_key.lower().strip()
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    catalog = get_model_catalog()
    if key not in catalog:
        raise ValueError(
            f"Modelo '{model_key}' no reconocido. Opciones válidas: {list(catalog.keys())}"
        )

    info = catalog[key]
    if not info["available"]:
        msg = f"El modelo '{key}' no está disponible actualmente."
        if info.get("notes"):
            msg += f" ({info['notes']})"
        raise RuntimeError(msg)

    ruta = Path(info["path"])

    if key in ("svm", "lightgbm"):
        model = cargar_modelo(ruta)
    elif key == "cnn1d":
        model = cargar_modelo_cnn(ruta)
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

    probs = {
        str(clase): float(prob)
        for clase, prob in zip(clases, probabilidades)
    }

    # Asegurar que todas las clases canónicas estén presentes en el diccionario
    for c in CLASES_CANONICAS:
        if c not in probs:
            probs[c] = 0.0

    return etapa, probs


# ── Inferencia CNN 1D ─────────────────────────────────────────────────────────
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


def preparar_tensor_cnn(raw, inicio: float, duracion: float = 30.0, n_muestras: int = 3000):
    """
    Extrae los 3 canales crudos, ajusta a 3000 muestras a 100 Hz,
    aplica normalización z-score por canal y retorna tensor (1, 3, 3000).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    fs = raw.info["sfreq"]
    start = int(inicio * fs)
    stop = int((inicio + duracion) * fs)

    for canal in CANALES_ML:
        if canal not in raw.ch_names:
            raise ValueError(f"Canal no encontrado en EDF: {canal}")

    data = raw.get_data(picks=CANALES_ML, start=start, stop=stop).astype(np.float32)

    # Ajuste de tamaño exacto si difiere por redondeo o sample rate
    if data.shape[1] != n_muestras:
        from scipy.signal import resample
        data = resample(data, n_muestras, axis=1).astype(np.float32)

    # Normalización z-score por canal
    for ch in range(data.shape[0]):
        mu = float(data[ch].mean())
        sigma = float(data[ch].std())
        if sigma > 0:
            data[ch] = (data[ch] - mu) / sigma

    # Expandir dimensión de batch: (1, 3, 3000)
    return torch.from_numpy(data).unsqueeze(0)


def predecir_cnn(modelo, x_tensor) -> Tuple[str, Dict[str, float]]:
    """Predice la etapa de sueño y distribución de probabilidad usando CNN 1D."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch (torch) no está disponible en este entorno.")

    with torch.no_grad():
        logits = modelo(x_tensor)
        probabilidades = torch.softmax(logits, dim=1)[0].cpu().numpy()

    i_max = int(probabilidades.argmax())
    etapa = CLASES_CANONICAS[i_max]

    probs = {
        clase: float(prob)
        for clase, prob in zip(CLASES_CANONICAS, probabilidades)
    }

    return etapa, probs


# ── Función Unificada de Inferencia ───────────────────────────────────────────
def ejecutar_inferencia_modelo(
    model_key: str,
    raw,
    inicio: float,
    duracion: float = 30.0,
    features_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Ejecuta la inferencia para un modelo específico (svm, lightgbm o cnn1d).
    Reutiliza features_df si ya fue calculado previamente para optimizar tiempo.
    """
    key = model_key.lower().strip()
    modelo = obtener_modelo(key)

    if key in ("svm", "lightgbm"):
        if features_df is None:
            features_df = preparar_features(raw, inicio=inicio, duracion=duracion)
        etapa, probs = predecir(modelo, features_df)
    elif key == "cnn1d":
        x_tensor = preparar_tensor_cnn(raw, inicio=inicio, duracion=duracion)
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