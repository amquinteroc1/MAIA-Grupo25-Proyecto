"""
crear_dataset_tesis.py
Pipeline de preparación de datos para el Proyecto de Grado.
Implementa el protocolo experimental unificado (Entrega 1, sección 2).

Cambios respecto al pipeline del microproyecto (crear_dataset_cnn.py):
1. SC y ST se separan antes de cualquier split.
2. SC se divide en train/val/test (64/16/20) por sujeto.
3. Normalización poblacional: params estimados solo en SC-train, aplicados a todo.
4. Trimming de Wake prolongado: margen de 30 min antes/después del tiempo en cama.
5. Inventario de registros con trazabilidad completa.
6. Secuencias (5,3,3000) sin padding — épocas de borde se excluyen.

Uso:
    python -m src.data_tesis.crear_dataset_tesis
    python -m src.data_tesis.crear_dataset_tesis --fase e0      # solo tensores E0
    python -m src.data_tesis.crear_dataset_tesis --fase seq     # solo secuencias (requiere E0)
    python -m src.data_tesis.crear_dataset_tesis --fase todo    # completo (default)
"""

import os

for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import csv
import gc
import glob
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import mne
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("crear_dataset_tesis")

# ── Constantes ────────────────────────────────────────────────────────────────
CANALES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
FS = 100
EPOCH_SEC = 30
N_MUESTRAS = FS * EPOCH_SEC  # 3000
SEQ_LEN = 5
SPLIT_SEED = 42
TRIM_MARGIN_MIN = 30  # minutos de Wake a conservar antes/después del sueño

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
EXCLUIR = {"Sleep stage M", "Sleep stage ?", "Movement time"}
CLASE_NOMBRES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

ROOT = Path(__file__).resolve().parents[2]
RAW_SC = ROOT / "data" / "raw" / "sleep-cassette"
RAW_ST = ROOT / "data" / "raw" / "sleep-telemetry"
OUT_E0 = ROOT / "data" / "processed" / "e0"
OUT_SEQ = ROOT / "data" / "processed" / "seq"
OUT_META = ROOT / "data" / "processed" / "meta"


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE LECTURA (reutilizadas del microproyecto con trimming añadido)
# ══════════════════════════════════════════════════════════════════════════════

def encontrar_pares(data_dir):
    """Busca pares PSG/Hypnogram en un directorio. Retorna lista de (psg, hyp, subject_id)."""
    pares = []
    d_path = Path(data_dir)
    if not d_path.exists():
        return pares
    for psg in sorted(glob.glob(str(d_path / "*-PSG.edf"))):
        base = psg.replace("-PSG.edf", "")
        hyp_candidates = glob.glob(base[:-2] + "*-Hypnogram.edf")
        if hyp_candidates:
            subject_id = Path(psg).stem[:7]
            pares.append((psg, hyp_candidates[0], subject_id))
    return pares


def leer_sujeto(psg_path, hyp_path, trim_wake=True):
    """
    Lee y segmenta épocas de un sujeto.

    Retorna:
        epocas:    (N, 3, 3000) float32 — señal cruda en µV
        labels:    (N,) int64
        info:      dict con conteos de exclusiones para el inventario
    """
    info = {
        "epochs_raw": 0,
        "epochs_excluded_mq": 0,
        "epochs_excluded_trim": 0,
        "epochs_valid_e0": 0,
        "ch_fpz_cz": False,
        "ch_pz_oz": False,
        "ch_eog": False,
        "fs_verified": 0,
    }

    raw = None
    try:
        raw = mne.io.read_raw_edf(psg_path, include=CANALES, preload=True, verbose=False)

        # Verificar canales
        info["ch_fpz_cz"] = "EEG Fpz-Cz" in raw.ch_names
        info["ch_pz_oz"] = "EEG Pz-Oz" in raw.ch_names
        info["ch_eog"] = "EOG horizontal" in raw.ch_names
        info["fs_verified"] = int(raw.info["sfreq"])

        if any(ch not in raw.ch_names for ch in CANALES):
            return None, None, info

        ann = mne.read_annotations(hyp_path)
        datos = np.asarray(raw.get_data(picks=CANALES), dtype=np.float32)
        total_muestras = datos.shape[1]

        # Cerrar MNE
        if hasattr(raw, "close"):
            raw.close()
        raw = None

        # Extraer todas las épocas válidas con sus intervalos
        all_intervals = []  # (idx_ini, idx_fin, label)
        excluded_mq = 0

        for a in ann:
            desc = a["description"]
            if desc in EXCLUIR:
                dur_s = a["duration"]
                excluded_mq += int(dur_s // EPOCH_SEC)
                continue
            if desc not in LABEL_MAP:
                continue

            label = LABEL_MAP[desc]
            inicio_s = a["onset"]
            dur_s = a["duration"]

            t = inicio_s
            while t + EPOCH_SEC <= inicio_s + dur_s:
                idx_ini = int(t * FS)
                idx_fin = idx_ini + N_MUESTRAS
                if idx_fin > total_muestras:
                    break
                all_intervals.append((idx_ini, idx_fin, label))
                t += EPOCH_SEC

        info["epochs_raw"] = len(all_intervals) + excluded_mq
        info["epochs_excluded_mq"] = excluded_mq

        if not all_intervals:
            del datos
            return None, None, info

        # ── Trimming de Wake prolongado ───────────────────────────────────
        if trim_wake:
            labels_tmp = [iv[2] for iv in all_intervals]

            # Encontrar primera y última época no-Wake
            first_sleep = None
            last_sleep = None
            for i, lbl in enumerate(labels_tmp):
                if lbl != 0:  # no Wake
                    if first_sleep is None:
                        first_sleep = i
                    last_sleep = i

            if first_sleep is not None:
                margin_epochs = (TRIM_MARGIN_MIN * 60) // EPOCH_SEC  # 60 épocas
                trim_start = max(0, first_sleep - margin_epochs)
                trim_end = min(len(all_intervals), last_sleep + margin_epochs + 1)

                n_before_trim = len(all_intervals)
                all_intervals = all_intervals[trim_start:trim_end]
                info["epochs_excluded_trim"] = n_before_trim - len(all_intervals)

        if not all_intervals:
            del datos
            return None, None, info

        # ── Segmentar ────────────────────────────────────────────────────
        n_epocas = len(all_intervals)
        epocas = np.empty((n_epocas, len(CANALES), N_MUESTRAS), dtype=np.float32)
        labels = np.empty(n_epocas, dtype=np.int64)

        for i, (ini, fin, lbl) in enumerate(all_intervals):
            seg = datos[:, ini:fin]
            if seg.shape[1] < N_MUESTRAS:
                pad = np.zeros((seg.shape[0], N_MUESTRAS - seg.shape[1]), dtype=np.float32)
                epocas[i] = np.concatenate([seg, pad], axis=1)
            else:
                epocas[i] = seg
            labels[i] = lbl

        info["epochs_valid_e0"] = n_epocas

        del datos, all_intervals
        return epocas, labels, info

    finally:
        if raw is not None:
            if hasattr(raw, "close"):
                raw.close()


# ══════════════════════════════════════════════════════════════════════════════
# PARTICIONAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def split_sc_sujetos(sujetos_sc):
    """
    Divide sujetos SC en train/val/test (64/16/20).
    Retorna dict: {subject_id: 'train'|'val'|'test'}
    """
    sujetos = sorted(sujetos_sc)
    arr = np.array(sujetos)

    # Paso 1: separar test (20%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    idx_trainval, idx_test = next(gss1.split(arr, arr, groups=arr))

    # Paso 2: del 80% restante, separar val (20% del 80% = 16% del total)
    arr_trainval = arr[idx_trainval]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    idx_train, idx_val = next(gss2.split(arr_trainval, arr_trainval, groups=arr_trainval))

    asignacion = {}
    for i in idx_test:
        asignacion[arr[i]] = "test"
    for i in idx_train:
        asignacion[arr_trainval[i]] = "train"
    for i in idx_val:
        asignacion[arr_trainval[i]] = "val"

    return asignacion


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN POBLACIONAL
# ══════════════════════════════════════════════════════════════════════════════

def calcular_norm_params(pares_train):
    """
    Calcula media y std por canal usando solo los sujetos de SC-train.
    Streaming (Welford online) para no cargar todo en RAM.
    """
    n_total = np.zeros(len(CANALES), dtype=np.float64)
    suma = np.zeros(len(CANALES), dtype=np.float64)
    suma_sq = np.zeros(len(CANALES), dtype=np.float64)

    for psg, hyp, sid in pares_train:
        epocas, labels, _ = leer_sujeto(psg, hyp, trim_wake=True)
        if epocas is None:
            continue

        for ch in range(len(CANALES)):
            canal = epocas[:, ch, :].astype(np.float64)
            n_total[ch] += canal.size
            suma[ch] += canal.sum()
            suma_sq[ch] += (canal ** 2).sum()

        del epocas, labels
        gc.collect()

    media = suma / n_total
    std = np.sqrt(suma_sq / n_total - media ** 2)

    return media, std


def aplicar_normalizacion(epocas, media, std):
    """Aplica z-score con parámetros externos (de SC-train). In-place."""
    for ch in range(epocas.shape[1]):
        if std[ch] > 0:
            epocas[:, ch, :] -= media[ch]
            epocas[:, ch, :] /= std[ch]
    return epocas


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE TENSORES E0
# ══════════════════════════════════════════════════════════════════════════════

def generar_tensores_e0(pares_por_particion, media, std, inventario):
    """
    Genera X, y, groups para cada partición (train/val/test/st).
    Escribe archivos .npy en data/processed/e0/.
    """
    for particion, pares in pares_por_particion.items():
        log.info("Generando tensores E0 para partición: %s (%d registros)", particion, len(pares))

        all_epocas = []
        all_labels = []
        all_groups = []

        for psg, hyp, sid in pares:
            epocas, labels, info = leer_sujeto(psg, hyp, trim_wake=True)

            # Actualizar inventario
            inv_row = inventario[sid]
            inv_row.update(info)

            if epocas is None:
                continue

            # Normalizar con params de SC-train
            epocas = aplicar_normalizacion(epocas, media, std)

            # Distribución de clases para inventario
            for c in range(5):
                inv_row[f"count_{CLASE_NOMBRES[c].lower()}"] = int((labels == c).sum())

            all_epocas.append(epocas)
            all_labels.append(labels)
            all_groups.append(np.full(len(labels), hash(sid) % (2**31), dtype=np.int64))

            del epocas, labels

        if not all_epocas:
            log.warning("Partición %s sin épocas válidas", particion)
            continue

        X = np.concatenate(all_epocas, axis=0)
        y = np.concatenate(all_labels, axis=0)
        g = np.concatenate(all_groups, axis=0)

        suffix = particion  # train, val, test, st
        np.save(OUT_E0 / f"X_{suffix}.npy", X)
        np.save(OUT_E0 / f"y_{suffix}.npy", y)
        np.save(OUT_E0 / f"groups_{suffix}.npy", g)

        log.info("  %s: %d épocas | shape X: %s", particion, len(y), X.shape)
        log.info("  Distribución: %s", {CLASE_NOMBRES[i]: int((y == i).sum()) for i in range(5)})

        del X, y, g, all_epocas, all_labels, all_groups
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN DE SECUENCIAS E1/E2
# ══════════════════════════════════════════════════════════════════════════════

def construir_secuencias_registro(epocas, labels, seq_len=SEQ_LEN):
    """
    Construye secuencias de seq_len épocas consecutivas para un solo registro.
    Sin padding — las épocas de borde se excluyen.
    La etiqueta corresponde a la época central (índice seq_len//2).

    Retorna:
        X_seq:       (M, seq_len, C, T) float32
        y_seq:       (M,) int64
        eligible_idx: (M,) int64 — índices de las épocas centrales en el array E0 del registro
    """
    pad = seq_len // 2  # 2 para seq_len=5
    n = len(epocas)

    if n < seq_len:
        return None, None, None

    M = n - 2 * pad  # épocas que tienen contexto completo
    X_seq = np.empty((M, seq_len, epocas.shape[1], epocas.shape[2]), dtype=np.float32)
    y_seq = np.empty(M, dtype=np.int64)
    eligible_idx = np.arange(pad, n - pad, dtype=np.int64)

    for i in range(M):
        start = i
        end = i + seq_len
        X_seq[i] = epocas[start:end]
        y_seq[i] = labels[i + pad]  # época central

    return X_seq, y_seq, eligible_idx


def generar_tensores_seq(pares_por_particion, inventario):
    """
    Genera tensores de secuencias (5,3,3000) para cada partición.
    Lee los tensores E0 ya normalizados y construye secuencias por registro.
    """
    for particion, pares in pares_por_particion.items():
        log.info("Generando secuencias para partición: %s", particion)

        suffix = particion
        X_e0 = np.load(OUT_E0 / f"X_{suffix}.npy")
        y_e0 = np.load(OUT_E0 / f"y_{suffix}.npy")
        g_e0 = np.load(OUT_E0 / f"groups_{suffix}.npy")

        # Agrupar por sujeto (grupo)
        unique_groups = np.unique(g_e0)

        all_seq_X = []
        all_seq_y = []
        all_seq_g = []
        all_eligible = []
        total_excluded_border = 0

        cursor = 0
        for gid in unique_groups:
            mask = g_e0 == gid
            idx_in_e0 = np.where(mask)[0]
            epocas_reg = X_e0[mask]
            labels_reg = y_e0[mask]

            X_s, y_s, elig = construir_secuencias_registro(epocas_reg, labels_reg)

            if X_s is None:
                total_excluded_border += len(epocas_reg)
                continue

            n_excluded = len(epocas_reg) - len(X_s)
            total_excluded_border += n_excluded

            all_seq_X.append(X_s)
            all_seq_y.append(y_s)
            all_seq_g.append(np.full(len(y_s), gid, dtype=np.int64))
            # Los eligible indices son relativos al array E0 global
            all_eligible.append(idx_in_e0[elig])

        if not all_seq_X:
            log.warning("Partición %s sin secuencias válidas", particion)
            continue

        X_seq = np.concatenate(all_seq_X, axis=0)
        y_seq = np.concatenate(all_seq_y, axis=0)
        g_seq = np.concatenate(all_seq_g, axis=0)
        eligible = np.concatenate(all_eligible, axis=0)

        np.save(OUT_SEQ / f"X_seq_{suffix}.npy", X_seq)
        np.save(OUT_SEQ / f"y_seq_{suffix}.npy", y_seq)
        np.save(OUT_SEQ / f"groups_seq_{suffix}.npy", g_seq)
        np.save(OUT_SEQ / f"eligible_indices_{suffix}.npy", eligible)

        log.info("  %s: %d secuencias (de %d épocas E0) | excluidas borde: %d",
                 particion, len(y_seq), len(y_e0), total_excluded_border)
        log.info("  Shape X_seq: %s", X_seq.shape)

        # Actualizar inventario con exclusiones de borde agregadas
        # (se registra en split_info, no por sujeto individual — es por partición)

        del X_e0, y_e0, g_e0, X_seq, y_seq, g_seq, eligible
        del all_seq_X, all_seq_y, all_seq_g, all_eligible
        gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# INVENTARIO Y METADATOS
# ══════════════════════════════════════════════════════════════════════════════

def guardar_inventario(inventario, path):
    """Guarda inventory_records.csv."""
    fieldnames = [
        "subject_id", "subset", "partition", "psg_file", "hyp_file",
        "ch_fpz_cz", "ch_pz_oz", "ch_eog", "fs_verified",
        "epochs_raw", "epochs_excluded_mq", "epochs_excluded_trim",
        "epochs_valid_e0", "epochs_excluded_border", "epochs_valid_seq",
        "count_wake", "count_n1", "count_n2", "count_n3", "count_rem",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(inventario.keys()):
            row = inventario[sid]
            writer.writerow(row)
    log.info("Inventario guardado: %s (%d registros)", path, len(inventario))


def guardar_split_info(asignacion_sc, sujetos_st, media, std):
    """Guarda split_info.json con toda la metadata del protocolo."""
    # Leer conteos de los tensores generados
    info = {
        "protocolo": "Protocolo experimental unificado — Entrega 1, sección 2",
        "split_seed": SPLIT_SEED,
        "split_proporciones": {"train": 0.64, "val": 0.16, "test": 0.20, "st": 1.0},
        "trim_wake_margin_min": TRIM_MARGIN_MIN,
        "canales": CANALES,
        "sample_rate_hz": FS,
        "epoch_duration_s": EPOCH_SEC,
        "n_muestras_epoca": N_MUESTRAS,
        "seq_len": SEQ_LEN,
        "norm_params_file": "norm_params.npz",
        "norm_media_por_canal": media.tolist(),
        "norm_std_por_canal": std.tolist(),
        "sujetos": {},
        "conteos_e0": {},
        "conteos_seq": {},
    }

    # Sujetos por partición
    particiones = {"train": [], "val": [], "test": [], "st": []}
    for sid, part in asignacion_sc.items():
        particiones[part].append(sid)
    for sid in sorted(sujetos_st):
        particiones["st"].append(sid)
    info["sujetos"] = {k: sorted(v) for k, v in particiones.items()}
    info["n_sujetos"] = {k: len(v) for k, v in particiones.items()}

    # Conteos desde tensores E0
    for suffix in ["train", "val", "test", "st"]:
        y_path = OUT_E0 / f"y_{suffix}.npy"
        if y_path.exists():
            y = np.load(y_path)
            info["conteos_e0"][suffix] = {
                "total": int(len(y)),
                "por_clase": {CLASE_NOMBRES[i]: int((y == i).sum()) for i in range(5)},
            }

    # Conteos desde tensores seq
    for suffix in ["train", "val", "test", "st"]:
        y_path = OUT_SEQ / f"y_seq_{suffix}.npy"
        if y_path.exists():
            y = np.load(y_path)
            info["conteos_seq"][suffix] = {
                "total": int(len(y)),
                "por_clase": {CLASE_NOMBRES[i]: int((y == i).sum()) for i in range(5)},
            }

    # Épocas excluidas por borde (diferencia E0 - seq)
    info["exclusiones_borde"] = {}
    for suffix in ["train", "val", "test", "st"]:
        n_e0 = info["conteos_e0"].get(suffix, {}).get("total", 0)
        n_seq = info["conteos_seq"].get(suffix, {}).get("total", 0)
        if n_e0 > 0:
            info["exclusiones_borde"][suffix] = {
                "epochs_e0": n_e0,
                "epochs_seq": n_seq,
                "excluded_border": n_e0 - n_seq,
            }

    path = OUT_META / "split_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    log.info("split_info.json guardado: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline de datos — Proyecto de Grado")
    parser.add_argument("--fase", choices=["e0", "seq", "todo"], default="todo",
                        help="Qué fase ejecutar: e0, seq, o todo (default)")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Pipeline de datos — Proyecto de Grado")
    log.info("Fase: %s", args.fase)
    log.info("=" * 70)

    # ── 1. Encontrar pares ────────────────────────────────────────────────
    pares_sc = encontrar_pares(RAW_SC)
    pares_st = encontrar_pares(RAW_ST)
    log.info("Pares encontrados: %d SC + %d ST = %d total",
             len(pares_sc), len(pares_st), len(pares_sc) + len(pares_st))

    sujetos_sc = sorted(set(p[2] for p in pares_sc))
    sujetos_st = sorted(set(p[2] for p in pares_st))

    # ── 2. Particionar SC ─────────────────────────────────────────────────
    asignacion_sc = split_sc_sujetos(sujetos_sc)

    conteo_part = {"train": 0, "val": 0, "test": 0}
    for part in asignacion_sc.values():
        conteo_part[part] += 1
    log.info("Partición SC: train=%d, val=%d, test=%d | ST=%d (completo)",
             conteo_part["train"], conteo_part["val"], conteo_part["test"], len(sujetos_st))

    # Verificar no solapamiento
    sets = {"train": set(), "val": set(), "test": set()}
    for sid, part in asignacion_sc.items():
        sets[part].add(sid)
    assert not (sets["train"] & sets["val"]), "Leakage train-val"
    assert not (sets["train"] & sets["test"]), "Leakage train-test"
    assert not (sets["val"] & sets["test"]), "Leakage val-test"
    assert not (set(sujetos_st) & set(sujetos_sc)), "Leakage SC-ST"

    # Organizar pares por partición
    pares_por_particion = {"train": [], "val": [], "test": [], "st": []}
    for psg, hyp, sid in pares_sc:
        pares_por_particion[asignacion_sc[sid]].append((psg, hyp, sid))
    for psg, hyp, sid in pares_st:
        pares_por_particion["st"].append((psg, hyp, sid))

    # ── 3. Inicializar inventario ─────────────────────────────────────────
    inventario = {}
    for psg, hyp, sid in pares_sc:
        subset = "SC"
        inventario[sid] = {
            "subject_id": sid,
            "subset": subset,
            "partition": asignacion_sc[sid],
            "psg_file": Path(psg).name,
            "hyp_file": Path(hyp).name,
            "ch_fpz_cz": False, "ch_pz_oz": False, "ch_eog": False,
            "fs_verified": 0,
            "epochs_raw": 0, "epochs_excluded_mq": 0, "epochs_excluded_trim": 0,
            "epochs_valid_e0": 0, "epochs_excluded_border": 0, "epochs_valid_seq": 0,
            "count_wake": 0, "count_n1": 0, "count_n2": 0, "count_n3": 0, "count_rem": 0,
        }
    for psg, hyp, sid in pares_st:
        inventario[sid] = {
            "subject_id": sid,
            "subset": "ST",
            "partition": "st",
            "psg_file": Path(psg).name,
            "hyp_file": Path(hyp).name,
            "ch_fpz_cz": False, "ch_pz_oz": False, "ch_eog": False,
            "fs_verified": 0,
            "epochs_raw": 0, "epochs_excluded_mq": 0, "epochs_excluded_trim": 0,
            "epochs_valid_e0": 0, "epochs_excluded_border": 0, "epochs_valid_seq": 0,
            "count_wake": 0, "count_n1": 0, "count_n2": 0, "count_n3": 0, "count_rem": 0,
        }

    if args.fase in ("e0", "todo"):
        # ── 4. Calcular parámetros de normalización (solo SC-train) ───────
        log.info("Calculando parámetros de normalización sobre SC-train...")
        media, std = calcular_norm_params(pares_por_particion["train"])
        log.info("  Media por canal: %s", media)
        log.info("  Std por canal:   %s", std)
        np.savez(OUT_META / "norm_params.npz", media=media, std=std)
        log.info("  norm_params.npz guardado")

        # ── 5. Generar tensores E0 ────────────────────────────────────────
        generar_tensores_e0(pares_por_particion, media, std, inventario)

        # ── 6. Guardar inventario ─────────────────────────────────────────
        guardar_inventario(inventario, OUT_META / "inventory_records.csv")

    if args.fase in ("seq", "todo"):
        # Cargar norm_params si no están en memoria
        if args.fase == "seq":
            np_data = np.load(OUT_META / "norm_params.npz")
            media = np_data["media"]
            std = np_data["std"]

        # ── 7. Generar secuencias E1/E2 ──────────────────────────────────
        generar_tensores_seq(pares_por_particion, inventario)

    # ── 8. Guardar split_info.json ────────────────────────────────────────
    if args.fase == "seq":
        np_data = np.load(OUT_META / "norm_params.npz")
        media, std = np_data["media"], np_data["std"]
    guardar_split_info(asignacion_sc, sujetos_st, media, std)

    log.info("=" * 70)
    log.info("Pipeline completado. Artefactos en data/processed/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
