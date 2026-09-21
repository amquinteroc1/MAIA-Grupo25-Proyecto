"""
predict.py
Módulo de inferencia para clasificación de etapas de sueño en ventanas de 30s.
Soporta tanto el modelo Baseline (SVM) como Deep Learning (CNN 1D).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from src.data.features import extraer_features

# Configuración de canales estándar
CANALES_ML = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
CLASES_CANONICAS = ["Wake", "N1", "N2", "N3", "REM"]


# ── Inferencia SVM ────────────────────────────────────────────────────────────
def cargar_modelo(ruta):
    """Carga un modelo serializado con joblib (SVM)."""
    return joblib.load(Path(ruta))


def preparar_features(raw, inicio, duracion=30):
    """Extrae las 30 características espectrales/temporales para el modelo SVM."""
    fs = raw.info["sfreq"]
    start = int(inicio * fs)
    stop = int((inicio + duracion) * fs)

    fila = {}
    for canal in CANALES_ML:
        if canal not in raw.ch_names:
            raise ValueError(f"Canal no encontrado: {canal}")

        signal = raw.get_data(picks=[canal], start=start, stop=stop)[0]
        features = extraer_features(signal, fs)
        nombre_canal = canal.replace(" ", "_").replace("-", "_")

        for nombre_feature, valor in features.items():
            fila[f"{nombre_canal}_{nombre_feature}"] = valor

    return pd.DataFrame([fila])


def predecir(modelo, X):
    """Predice la etapa de sueño y probabilidades usando SVM."""
    probabilidades = modelo.predict_proba(X)[0]
    clases = modelo.classes_

    i_max = probabilidades.argmax()
    etapa = clases[i_max]

    probs = {
        clase: float(prob)
        for clase, prob in zip(clases, probabilidades)
    }

    return etapa, probs


# ── Inferencia CNN 1D ─────────────────────────────────────────────────────────
def cargar_modelo_cnn(ruta):
    """Carga los pesos del modelo PyTorch CNN 1D en modo evaluación."""
    from src.models.cnn1d import build_model

    ckpt = torch.load(Path(ruta), map_location="cpu", weights_only=False)
    dropout = ckpt.get("args", {}).get("dropout", 0.5) if isinstance(ckpt, dict) else 0.5
    model = build_model(n_channels=3, n_classes=5, dropout=dropout)

    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preparar_tensor_cnn(raw, inicio, duracion=30, n_muestras=3000):
    """
    Extrae los 3 canales crudos, ajusta a 3000 muestras a 100 Hz,
    aplica normalización z-score por canal y retorna tensor (1, 3, 3000).
    """
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


def predecir_cnn(modelo, x_tensor):
    """Predice la etapa de sueño y distribución de probabilidad usando CNN 1D."""
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