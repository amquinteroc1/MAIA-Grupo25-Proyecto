"""
config.py — Fuente única de verdad para los experimentos E0, E1 y E2.

Todos los scripts de entrenamiento importan desde aquí.
Ningún valor crítico debe estar hardcodeado en otro archivo.

Proyecto: Efecto del contexto temporal en la clasificación de estadios del sueño
Grupo 25 — Universidad de los Andes, 2026
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════════════════════

# Este archivo está en <repositorio>/src/config.py:
# parents[0] -> <repositorio>/src; parents[1] -> <repositorio>.
ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    # Datos regenerados con la partición corregida por persona.
    # E0: épocas individuales de 30 segundos.
    "data_e0": ROOT / "data" / "processed" / "e0",
    # E1 y E2: secuencias de cinco épocas realmente consecutivas.
    "data_seq": ROOT / "data" / "processed" / "seq",
    # Inventario, particiones y parámetros de normalización.
    "data_meta": ROOT / "data" / "processed" / "meta",

    # Artefactos de experimentos (modelos, predicciones, logs).
    "experiments": ROOT / "experiments",
    "mlruns": ROOT / "mlruns",
}

# ══════════════════════════════════════════════════════════════════════════════
# CLASES
# ══════════════════════════════════════════════════════════════════════════════

CLASSES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES = 5

# ══════════════════════════════════════════════════════════════════════════════
# ARQUITECTURA CNN 1D BASE (compartida entre E0, E1 y E2)
# Definida en el Informe de Avance, sección 3.2 — NO modificar.
# ══════════════════════════════════════════════════════════════════════════════

CNN_ARCH = {
    # Extractor convolucional — 4 bloques Conv1D + BN + ReLU + MaxPool
    "n_channels": 3,       # EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal
    "n_samples": 3000,     # 30 s × 100 Hz
    "filters": [32, 64, 128, 256],
    "kernels": [7, 5, 3, 3],
    "pools": [5, 5, 4, 2],
    # Resultado del extractor: 256 canales × 15 posiciones = 3840
    "flat_size": 3840,     # 256 * 15

    # Cabezal denso
    "fc_hidden": 512,
    "dropout": 0.5,
    "n_classes": N_CLASSES,
}

# ══════════════════════════════════════════════════════════════════════════════
# CABEZALES ESPECÍFICOS POR EXPERIMENTO
# ══════════════════════════════════════════════════════════════════════════════

E0_HEAD = {
    # Linear(3840 → 512) + ReLU + Dropout(0.5) + Linear(512 → 5)
    "in_features": CNN_ARCH["flat_size"],
    "hidden": CNN_ARCH["fc_hidden"],
    "out_features": N_CLASSES,
}

E1_HEAD = {
    # 5 embeddings de 512 concatenados → Linear(2560 → 5)
    "seq_len": 5,
    "embed_size": CNN_ARCH["fc_hidden"],
    "in_features": 5 * CNN_ARCH["fc_hidden"],
    "out_features": N_CLASSES,
}

E2_HEAD = {
    # 5 embeddings de 512 → GRU → Linear(gru_hidden → 5)
    "seq_len": 5,
    "embed_size": CNN_ARCH["fc_hidden"],
    "gru_hidden": 256,  # por definir tras corrida exploratoria — candidato inicial
    "gru_layers": 1,
    "out_features": N_CLASSES,
}

# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLO DE SEMILLAS Y REPETICIONES
# Sección 2.4 del Informe de Avance — inamovible.
# ══════════════════════════════════════════════════════════════════════════════

SEEDS = [42, 123, 2026]
SPLIT_SEED = 42
SEED_LABELS = {42: "A", 123: "B", 2026: "C"}

# ══════════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS DE ENTRENAMIENTO
# Estos aplican igual a E0, E1 y E2 (excepto donde se indique).
# ══════════════════════════════════════════════════════════════════════════════

TRAINING = {
    "batch_size": 128,

    # Optimizador Adam
    "lr": 1e-3,
    "weight_decay": 1e-4,

    # Scheduler ReduceLROnPlateau — monitorea val_f1_macro
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,

    # Early stopping — monitorea val_f1_macro
    # PROVISIONAL: ajustar tras corrida exploratoria de E0
    "max_epochs": 30,
    "early_stopping_patience": 10,

    "monitor_metric": "val_f1_macro",
    "class_weight_strategy": "balanced",
    "num_workers": 0,
    "pin_memory": True,
}

# ══════════════════════════════════════════════════════════════════════════════
# SEGUIMIENTO Y REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

MLFLOW = {
    "experiment_name": "sleep-staging-contexto-temporal",
    "run_name_template": "{experiment}-seed{seed}",
    "tracking_uri": f"sqlite:///{ROOT}/mlruns.db",
}

HF_REPO = "maia-grupo25/sleep-edfx-data"

# ══════════════════════════════════════════════════════════════════════════════
# SECUENCIA EXPERIMENTAL
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "E0": "CNN 1D por época — línea base",
    "E1": "Contexto por concatenación — 5 épocas",
    "E2": "Contexto con GRU — 5 épocas",
}

RUNS = [
    {"experiment": exp, "seed": seed, "label": SEED_LABELS[seed]}
    for exp in ["E0", "E1", "E2"]
    for seed in SEEDS
]
