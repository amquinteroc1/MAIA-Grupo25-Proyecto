"""
crear_dataset_cnn.py
Pipeline de datos para CNN 1D — clasificación de estadios de sueño.

Genera dos conjuntos de artefactos:
  MVP 1 — épocas individuales : X shape (N, 3, 3000)
  MVP 2 — secuencias de épocas: X shape (N, 5, 3, 3000)

NO modifica ni lee artefactos del pipeline SVM.
Fuente de datos: data/sleep-cassette/ y data/sleep-telemetry/
"""

import os
import json
import glob
import logging
import numpy as np
import mne
import mlflow
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# ── Configuración de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
CANALES        = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
FS             = 100          # Hz — frecuencia de muestreo
EPOCH_SEC      = 30           # segundos por época
N_MUESTRAS     = FS * EPOCH_SEC   # 3000 muestras por época
SEQ_LEN        = 5            # épocas de contexto para MVP 2
SPLIT_SEED     = 42
SPLIT_RATIO    = 0.2          # 80/20

# Mapeo Rechtschaffen & Kales → 5 clases AASM
LABEL_MAP = {
    "Sleep stage W":  0,   # Wake
    "Sleep stage 1":  1,   # N1
    "Sleep stage 2":  2,   # N2
    "Sleep stage 3":  3,   # N3
    "Sleep stage 4":  3,   # N3 (agrupa 3 y 4)
    "Sleep stage R":  4,   # REM
}
EXCLUIR = {"Sleep stage M", "Sleep stage ?", "Movement time"}

# Rutas
ROOT         = Path(__file__).resolve().parents[2]
DATA_DIRS    = [ROOT / "data" / "sleep-cassette",
                ROOT / "data" / "sleep-telemetry"]
OUT_DIR      = ROOT / "data_processed"
OUT_DIR.mkdir(exist_ok=True)


# ── Paso 1: encontrar pares PSG + Hypnogram ───────────────────────────────────
def encontrar_pares():
    """Retorna lista de tuplas (psg_path, hyp_path, subject_id)."""
    pares = []
    for data_dir in DATA_DIRS:
        psg_files = sorted(glob.glob(str(data_dir / "*-PSG.edf")))
        for psg in psg_files:
            base = psg.replace("-PSG.edf", "")
            # Hypnogram puede terminar en EC, EH, EJ, etc.
            hyp_candidates = glob.glob(base[:-2] + "*-Hypnogram.edf")
            if not hyp_candidates:
                log.warning(f"Sin hypnogram para {psg} — omitido")
                continue
            hyp = hyp_candidates[0]
            # ID de sujeto = primeros 7 caracteres del nombre de archivo
            subject_id = Path(psg).stem[:7]
            pares.append((psg, hyp, subject_id))
    log.info(f"Pares encontrados: {len(pares)}")
    return pares


# ── Paso 2: leer un par EDF y segmentar en épocas ────────────────────────────
def leer_sujeto(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, include=CANALES,
                               preload=True, verbose=False)

    disponibles = raw.ch_names
    for ch in CANALES:
        if ch not in disponibles:
            log.warning(f"Canal {ch} no encontrado en {psg_path} — omitido")
            return None, None

    ann = mne.read_annotations(hyp_path)
    datos = raw.get_data(picks=CANALES)   # (3, total_muestras)
    total_muestras = datos.shape[1]

    epocas_list = []
    labels_list = []

    for a in ann:
        descripcion = a["description"]

        # Descartar épocas no clasificables
        if descripcion in EXCLUIR:
            continue

        # Descartar si la etiqueta no está en el mapa
        if descripcion not in LABEL_MAP:
            continue

        label   = LABEL_MAP[descripcion]
        inicio_s = a["onset"]       # segundos
        dur_s    = a["duration"]    # segundos — puede ser mucho mayor a 30

        # Iterar en pasos de 30 segundos dentro de esta anotación
        t = inicio_s
        while t + EPOCH_SEC <= inicio_s + dur_s:
            inicio_muestra = int(t * FS)
            fin_muestra    = inicio_muestra + N_MUESTRAS

            # Verificar que no se sale del registro
            if fin_muestra > total_muestras:
                break

            segmento = datos[:, inicio_muestra:fin_muestra]  # (3, 3000)
            epocas_list.append(segmento.astype(np.float32))
            labels_list.append(label)

            t += EPOCH_SEC   # avanzar 30 segundos

    if len(epocas_list) == 0:
        return None, None

    return np.stack(epocas_list), np.array(labels_list, dtype=np.int64)


# ── Paso 3: normalización por sujeto y canal ─────────────────────────────────
def normalizar(epocas):
    """
    Normalización z-score por canal calculada sobre todo el registro del sujeto.
    epocas: (N, 3, 3000)
    Retorna: (N, 3, 3000) normalizado
    """
    for ch in range(epocas.shape[1]):
        señal = epocas[:, ch, :]          # (N, 3000)
        mu    = señal.mean()
        sigma = señal.std()
        if sigma > 0:
            epocas[:, ch, :] = (señal - mu) / sigma
    return epocas


# ── Paso 4: construir secuencias de 5 épocas (MVP 2) ─────────────────────────
def construir_secuencias(epocas, labels, seq_len=SEQ_LEN):
    """
    Construye ventanas deslizantes de seq_len épocas.
    La etiqueta corresponde a la época central.
    Padding con ceros en los bordes.

    epocas: (N, 3, 3000)
    Retorna:
        X_seq: (N, seq_len, 3, 3000)
        y_seq: (N,)  — mismas etiquetas, la central
    """
    N = len(epocas)
    pad = seq_len // 2  # 2 épocas de padding en cada extremo

    # Padding con ceros
    pad_shape  = (pad, epocas.shape[1], epocas.shape[2])
    pad_bloque = np.zeros(pad_shape, dtype=np.float32)
    epocas_pad = np.concatenate([pad_bloque, epocas, pad_bloque], axis=0)

    X_seq = np.stack([
        epocas_pad[i: i + seq_len]
        for i in range(N)
    ])  # (N, seq_len, 3, 3000)

    return X_seq, labels.copy()


# ── Paso 5: pipeline completo ─────────────────────────────────────────────────
def main():
    log.info("=== Iniciando pipeline CNN ===")

    pares = encontrar_pares()
    assert len(pares) > 0, "No se encontraron pares EDF"

    # Mapeo sujeto → ID