"""
test_loader_smoke.py
Smoke test del loader CNN — verifica un solo sujeto antes de correr los 197.
Ejecutar: python -m src.data.test_loader_smoke
"""

import numpy as np
import mne
import glob
from pathlib import Path
from src.data.crear_dataset_cnn import (
    CANALES, N_MUESTRAS, SEQ_LEN,
    leer_sujeto, normalizar, construir_secuencias
)

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "sleep-cassette"

def main():
    # Tomar el primer par disponible
    psg_files = sorted(glob.glob(str(DATA_DIR / "*-PSG.edf")))
    assert len(psg_files) > 0, "No hay archivos PSG"

    psg = psg_files[0]
    base = psg.replace("-PSG.edf", "")
    hyp_candidates = glob.glob(base[:-2] + "*-Hypnogram.edf")
    assert len(hyp_candidates) > 0, "No hay Hypnogram para el primer sujeto"
    hyp = hyp_candidates[0]

    print(f"\n{'='*60}")
    print(f"Sujeto de prueba:")
    print(f"  PSG : {Path(psg).name}")
    print(f"  HYP : {Path(hyp).name}")
    print(f"{'='*60}")

    # ── Paso 1: lectura ───────────────────────────────────────────────────────
    print("\n[1] Leyendo EDF...")
    epocas, labels = leer_sujeto(psg, hyp)
    assert epocas is not None, "ERROR: leer_sujeto retornó None"
    print(f"  ✓ Shape épocas    : {epocas.shape}  esperado (N, 3, 3000)")
    print(f"  ✓ Shape labels    : {labels.shape}")
    print(f"  ✓ dtype épocas    : {epocas.dtype}  esperado float32")
    print(f"  ✓ dtype labels    : {labels.dtype}  esperado int64")
    assert epocas.shape[1] == 3,      "ERROR: se esperaban 3 canales"
    assert epocas.shape[2] == N_MUESTRAS, f"ERROR: se esperaban {N_MUESTRAS} muestras"
    assert epocas.dtype == np.float32, "ERROR: dtype incorrecto en épocas"
    assert labels.dtype == np.int64,   "ERROR: dtype incorrecto en labels"

    # ── Paso 2: distribución de clases ───────────────────────────────────────
    print("\n[2] Distribución de clases:")
    clases = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    for i, nombre in clases.items():
        count = int((labels == i).sum())
        print(f"  {nombre:5s} ({i}): {count:5d} épocas")
    assert set(labels).issubset({0,1,2,3,4}), "ERROR: etiquetas fuera de rango"

    # ── Paso 3: NaN / Inf ─────────────────────────────────────────────────────
    print("\n[3] Verificando NaN e Inf...")
    assert not np.isnan(epocas).any(), "ERROR: hay NaN en las épocas"
    assert not np.isinf(epocas).any(), "ERROR: hay Inf en las épocas"
    print("  ✓ Sin NaN ni Inf")

    # ── Paso 4: normalización ─────────────────────────────────────────────────
    print("\n[4] Normalizando...")
    epocas_norm = normalizar(epocas.copy())
    for ch in range(3):
        mu    = epocas_norm[:, ch, :].mean()
        sigma = epocas_norm[:, ch, :].std()
        print(f"  Canal {ch} — media: {mu:.4f}  std: {sigma:.4f}  "
              f"(esperado ~0 y ~1)")
    assert not np.isnan(epocas_norm).any(), "ERROR: NaN tras normalización"

    # ── Paso 5: secuencias MVP 2 ──────────────────────────────────────────────
    print(f"\n[5] Construyendo secuencias de {SEQ_LEN} épocas (MVP 2)...")
    X_seq, y_seq = construir_secuencias(epocas_norm, labels)
    print(f"  ✓ Shape X_seq: {X_seq.shape}  esperado (N, {SEQ_LEN}, 3, 3000)")
    print(f"  ✓ Shape y_seq: {y_seq.shape}")
    assert X_seq.shape[1] == SEQ_LEN, "ERROR: seq_len incorrecto"
    assert X_seq.shape[2] == 3,       "ERROR: canales incorrectos en secuencia"
    assert X_seq.shape[3] == N_MUESTRAS, "ERROR: muestras incorrectas en secuencia"
    assert len(X_seq) == len(epocas_norm), \
        "ERROR: número de secuencias != número de épocas"

    # ── Mini batch ────────────────────────────────────────────────────────────
    print("\n[6] Verificando mini-batch (32 muestras)...")
    batch_X = X_seq[:32]
    batch_y = y_seq[:32]
    assert batch_X.shape == (32, SEQ_LEN, 3, N_MUESTRAS), \
        f"ERROR: shape batch incorrecto: {batch_X.shape}"
    assert batch_y.shape == (32,), "ERROR: shape labels batch incorrecto"
    print(f"  ✓ batch X: {batch_X.shape}")
    print(f"  ✓ batch y: {batch_y.shape}")
    print(f"  ✓ dtype X: {batch_X.dtype}")

    print(f"\n{'='*60}")
    print("  ✅  SMOKE TEST PASADO — loader listo para 197 sujetos")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()