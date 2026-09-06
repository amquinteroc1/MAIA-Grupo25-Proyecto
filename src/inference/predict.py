from pathlib import Path
import joblib
import pandas as pd

from src.data.features import extraer_features


# configuracion
CANALES_ML = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]


# cargar modelo
def cargar_modelo(ruta):
    return joblib.load(Path(ruta))


# preparar ventana
def preparar_features(raw, inicio, duracion=30):
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


# predecir etapa
def predecir(modelo, X):
    probabilidades = modelo.predict_proba(X)[0]
    clases = modelo.classes_

    i_max = probabilidades.argmax()
    etapa = clases[i_max]

    probs = {
        clase: float(prob)
        for clase, prob in zip(clases, probabilidades)
    }

    return etapa, probs