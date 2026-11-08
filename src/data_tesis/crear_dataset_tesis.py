"""Pipeline corregido de datos Sleep-EDF para la tesis E0, E1 y E2.

Protocolo:
- SC: train/val/test (64/16/20 aproximadamente) por PERSONA, seed=42.
- ST: dominio externo, sin utilizarlo para entrenar o ajustar parametros.
- Una persona puede tener varios registros/noches; nunca se reparten entre splits.
- E0: epocas de 30 s, 3 canales, 100 Hz.
- E1/E2: 5 epocas cronologicamente adyacentes del MISMO registro.
- La etiqueta de E1/E2 es la etapa de la epoca central.
- Normalizacion por canal estimada EXCLUSIVAMENTE en SC-train.
- Los grupos numericos identifican personas y son reproducibles entre procesos.

Archivos compatibles con los loaders de E0/E1/E2 existentes:
 data/processed/e0/X_{train,val,test,st}.npy
 data/processed/e0/y_{train,val,test,st}.npy
 data/processed/e0/groups_{train,val,test,st}.npy
 data/processed/seq/X_seq_{train,val,test,st}.npy
 data/processed/seq/y_seq_{train,val,test,st}.npy
 data/processed/seq/groups_seq_{train,val,test,st}.npy
 data/processed/seq/eligible_indices_{train,val,test,st}.npy
 data/processed/meta/norm_params.npz
 data/processed/meta/inventory_records.csv
 data/processed/meta/split_info.json

Metadatos nuevos para auditar la continuidad y el origen:
 data/processed/e0/record_ids_{split}.npy
 data/processed/e0/epoch_starts_{split}.npy
 data/processed/meta/group_mappings.json

Uso desde la raiz del repositorio:
 python -m src.data_tesis.crear_dataset_tesis --fase todo
 python -m src.data_tesis.crear_dataset_tesis --fase e0
 python -m src.data_tesis.crear_dataset_tesis --fase seq

Dependencias: mne, numpy, scikit-learn.
No ejecutar el pipeline con datos anteriores sin regenerar E0 y las secuencias.
"""

import os

for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import gc
import json
import logging
import re
import sys
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("crear_dataset_tesis")

CANALES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
FS = 100
EPOCH_SEC = 30
N_MUESTRAS = FS * EPOCH_SEC
SEQ_LEN = 5
SPLIT_SEED = 42
TRIM_MARGIN_MIN = 30
PROTOCOL_VERSION = "tesis-subject-exclusive-chronological-v2"
PARTICIONES = ("train", "val", "test", "st")

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

# El archivo PSG de Sleep-EDF tiene el prefijo del registro (por ej. SC4001E0).
# Los 5 primeros caracteres identifican la persona real (por ej. SC400).
_ID_RE = re.compile(r"^(SC|ST)[A-Za-z0-9]{3}[A-Za-z0-9]+$")


def ids_del_registro(psg_path):
    """Devuelve (record_id completo, subject_id de 5 caracteres)."""
    filename = Path(psg_path).name
    if not filename.endswith("-PSG.edf"):
        raise ValueError(f"Archivo PSG no reconocido: {filename}")
    record_id = filename[:-len("-PSG.edf")]
    if not _ID_RE.fullmatch(record_id):
        raise ValueError(f"Identificador Sleep-EDF inesperado: {record_id}")
    return record_id, record_id[:5]


def encontrar_pares(data_dir):
    """Lista de (psg, hyp, record_id); record_id NO es subject_id."""
    d_path = Path(data_dir)
    if not d_path.is_dir():
        return []
    pares = []
    registros_vistos = set()
    for psg in sorted(d_path.glob("*-PSG.edf")):
        record_id, _ = ids_del_registro(psg)
        if record_id in registros_vistos:
            raise ValueError(f"Registro PSG duplicado: {record_id}")
        registros_vistos.add(record_id)
        # Sleep-EDF: el ultimo par de caracteres del PSG puede ser distinto
        # del Hypnogram; se emparejan por prefijo comun (como en el script base).
        prefix = record_id[:-2]
        candidatos = sorted(d_path.glob(f"{prefix}*-Hypnogram.edf"))
        if len(candidatos) != 1:
            raise ValueError(
                f"Se esperaba exactamente 1 hipnograma para {psg.name}; "
                f"se encontraron {len(candidatos)}: {[x.name for x in candidatos]}"
            )
        pares.append((str(psg), str(candidatos[0]), record_id))
    return pares


def _intervalos_validos(raw, hyp_path, trim_wake):
    """Devuelve (starts, labels, info) sin extraer las senales completas."""
    ch_names = raw.ch_names
    info = {
        "epochs_raw": 0,
        "epochs_excluded_mq": 0,
        "epochs_excluded_trim": 0,
        "epochs_excluded_outside": 0,
        "epochs_valid_e0": 0,
        "ch_fpz_cz": "EEG Fpz-Cz" in ch_names,
        "ch_pz_oz": "EEG Pz-Oz" in ch_names,
        "ch_eog": "EOG horizontal" in ch_names,
        "fs_verified": float(raw.info["sfreq"]),
    }
    faltan = [ch for ch in CANALES if ch not in ch_names]
    if faltan:
        raise ValueError(f"Faltan canales {faltan} en PSG")
    if not np.isclose(raw.info["sfreq"], FS, rtol=0, atol=1e-7):
        raise ValueError(f"Frecuencia esperada {FS} Hz, recibida {raw.info['sfreq']} Hz")

    annotations = mne.read_annotations(str(hyp_path))
    starts = []
    labels = []
    excluidos = 0
    fuera = 0
    for ann in annotations:
        desc = str(ann["description"])
        duration = float(ann["duration"])
        onset = float(ann["onset"])
        if desc in EXCLUIR:
            excluidos += int(np.floor(duration / EPOCH_SEC + 1e-9))
            continue
        if desc not in LABEL_MAP:
            log.warning("Anotacion ignorada en %s: %s", Path(hyp_path).name, desc)
            continue
        # Evita errores de precision acumulados al sumar 30 segundos repetidamente.
        n_epocas = int(np.floor(duration / EPOCH_SEC + 1e-9))
        for i in range(n_epocas):
            start = int(round((onset + i * EPOCH_SEC) * FS))
            if start < 0 or start + N_MUESTRAS > raw.n_times:
                fuera += 1
                continue
            starts.append(start)
            labels.append(LABEL_MAP[desc])

    info["epochs_excluded_mq"] = excluidos
    info["epochs_excluded_outside"] = fuera
    info["epochs_raw"] = len(starts) + excluidos + fuera
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), info

    starts = np.asarray(starts, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(starts, kind="stable")
    starts, labels = starts[order], labels[order]
    if len(starts) > 1 and np.any(np.diff(starts) < N_MUESTRAS):
        raise ValueError(
            f"Anotaciones repetidas o solapadas en {Path(hyp_path).name}; "
            "se detiene para evitar epocas ambiguas"
        )

    if trim_wake:
        sleep_idx = np.flatnonzero(labels != 0)
        if len(sleep_idx):
            margin = TRIM_MARGIN_MIN * 60  # segundos REALES, no filas depuradas
            first_start = starts[sleep_idx[0]]
            last_start = starts[sleep_idx[-1]]
            lo = first_start - margin * FS
            hi = last_start + margin * FS
            keep = (starts >= lo) & (starts <= hi)
            info["epochs_excluded_trim"] = int((~keep).sum())
            starts, labels = starts[keep], labels[keep]

    info["epochs_valid_e0"] = int(len(labels))
    return starts, labels, info


def leer_sujeto(psg_path, hyp_path, trim_wake=True, devolver_inicios=False,
                solo_metadata=False):
    """Lee una noche de Sleep-EDF.

    Conserva la firma antigua (epocas, labels, info) por compatibilidad con
    test_smoke_tesis.py. Con devolver_inicios=True agrega starts (en muestras)
    antes de info. `solo_metadata=True` evita cargar senales y se usa al contar.

    NOTA: MNE entrega senales en unidades SI (voltios), no en microvoltios.
    El z-score poblacional posterior produce datos adimensionales.
    """
    raw = mne.io.read_raw_edf(
        str(psg_path), include=CANALES, preload=not solo_metadata, verbose=False
    )
    try:
        starts, labels, info = _intervalos_validos(raw, hyp_path, trim_wake)
        epocas = None
        if not solo_metadata and len(starts):
            # Obtener datos de una noche, sin cargar todos los registros en RAM.
            datos = np.asarray(raw.get_data(picks=CANALES), dtype=np.float32)
            epocas = np.empty(
                (len(starts), len(CANALES), N_MUESTRAS), dtype=np.float32
            )
            for i, start in enumerate(starts):
                epocas[i] = datos[:, start:start + N_MUESTRAS]
            del datos
            if not np.isfinite(epocas).all():
                raise ValueError(f"Senales contienen NaN o Inf: {psg_path}")
        if devolver_inicios:
            return epocas, labels, starts, info
        return epocas, labels, info
    finally:
        raw.close()


def split_sc_sujetos(sujetos_sc):
    """Particion 64/16/20 sobre PERSONAS de SC, nunca sobre noches."""
    arr = np.asarray(sorted(set(sujetos_sc)))
    if len(arr) < 10 or not all(str(x).startswith("SC") and len(x) == 5 for x in arr):
        raise ValueError("split_sc_sujetos requiere >=10 IDs SC de persona (5 chars)")
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    idx_tv, idx_test = next(gss_test.split(arr, groups=arr))
    arr_tv = arr[idx_tv]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SPLIT_SEED)
    idx_train, idx_val = next(gss_val.split(arr_tv, groups=arr_tv))
    assignment = {str(arr[i]): "test" for i in idx_test}
    assignment.update({str(arr_tv[i]): "train" for i in idx_train})
    assignment.update({str(arr_tv[i]): "val" for i in idx_val})
    assert len(assignment) == len(arr)
    assert set(assignment.values()) == {"train", "val", "test"}
    return assignment


def _pares_particiones(pares_sc, pares_st, asignacion):
    pares = {p: [] for p in PARTICIONES}
    for psg, hyp, record_id in pares_sc:
        subject_id = record_id[:5]
        pares[asignacion[subject_id]].append((psg, hyp, record_id))
    for psg, hyp, record_id in pares_st:
        pares["st"].append((psg, hyp, record_id))
    return pares


def _mappings(pares_sc, pares_st):
    all_records = sorted(set(row[2] for row in pares_sc + pares_st))
    if len(all_records) != len(pares_sc) + len(pares_st):
        raise ValueError("Hay records duplicados en los conjuntos SC/ST")
    all_subjects = sorted(set(record_id[:5] for record_id in all_records))
    # IDs numericos consecutivos, reproducibles con el mismo inventario.
    subj_group = {sid: i + 1 for i, sid in enumerate(all_subjects)}
    rec_group = {rid: i + 1 for i, rid in enumerate(all_records)}
    return subj_group, rec_group


def verificar_particion_por_persona(pares_por_particion):
    """Se ejecuta antes de cualquier estadistica o tensor."""
    sets = {
        p: {rid[:5] for _, _, rid in pares_por_particion[p]}
        for p in PARTICIONES
    }
    for i, p in enumerate(PARTICIONES):
        for q in PARTICIONES[i + 1:]:
            compartidos = sets[p].intersection(sets[q])
            if compartidos:
                raise AssertionError(
                    f"LEAKAGE {p}-{q}: {sorted(compartidos)}"
                )
    return sets


def calcular_norm_params(pares_train):
    """Welford combinado por canal, SOLO para SC-train."""
    n_total = np.zeros(len(CANALES), dtype=np.int64)
    mean = np.zeros(len(CANALES), dtype=np.float64)
    m2 = np.zeros(len(CANALES), dtype=np.float64)
    for psg, hyp, record_id in pares_train:
        epocas, _, _ = leer_sujeto(psg, hyp, trim_wake=True)
        if epocas is None:
            log.warning("Sin epocas validas para normalizacion: %s", record_id)
            continue
        for ch in range(len(CANALES)):
            vec = epocas[:, ch, :].astype(np.float64, copy=False).reshape(-1)
            nb = int(vec.size)
            mb = float(vec.mean())
            m2b = float(np.square(vec - mb).sum(dtype=np.float64))
            na = int(n_total[ch])
            if na == 0:
                mean[ch], m2[ch], n_total[ch] = mb, m2b, nb
            else:
                delta = mb - mean[ch]
                nt = na + nb
                mean[ch] += delta * nb / nt
                m2[ch] += m2b + delta ** 2 * na * nb / nt
                n_total[ch] = nt
        del epocas
        gc.collect()
    if np.any(n_total == 0):
        raise ValueError("No hay muestras SC-train suficientes para normalizacion")
    std = np.sqrt(m2 / n_total)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("Desviacion estandar no positiva o no finita")
    return mean, std


def aplicar_normalizacion(epocas, media, std):
    """Normalizacion in-place, z-score global por canal."""
    for ch in range(len(CANALES)):
        epocas[:, ch, :] -= np.float32(media[ch])
        epocas[:, ch, :] /= np.float32(std[ch])
    return epocas


def _fila_inventario(psg, hyp, record_id, subset, partition, subj_group, rec_group):
    sid = record_id[:5]
    row = {
        "subject_id": sid,
        "record_id": record_id,
        "subject_group_id": subj_group[sid],
        "record_group_id": rec_group[record_id],
        "subset": subset,
        "partition": partition,
        "psg_file": Path(psg).name,
        "hyp_file": Path(hyp).name,
        "ch_fpz_cz": False,
        "ch_pz_oz": False,
        "ch_eog": False,
        "fs_verified": 0,
        "epochs_raw": 0,
        "epochs_excluded_mq": 0,
        "epochs_excluded_trim": 0,
        "epochs_excluded_outside": 0,
        "epochs_valid_e0": 0,
        "epochs_excluded_border": 0,
        "epochs_excluded_discontinuity": 0,
        "epochs_valid_seq": 0,
    }
    for cls in CLASE_NOMBRES.values():
        row[f"count_{cls.lower()}"] = 0
    return row


INVENTORY_COLUMNS = [
    "subject_id", "record_id", "subject_group_id", "record_group_id",
    "subset", "partition", "psg_file", "hyp_file",
    "ch_fpz_cz", "ch_pz_oz", "ch_eog", "fs_verified",
    "epochs_raw", "epochs_excluded_mq", "epochs_excluded_trim",
    "epochs_excluded_outside", "epochs_valid_e0",
    "epochs_excluded_border", "epochs_excluded_discontinuity", "epochs_valid_seq",
    "count_wake", "count_n1", "count_n2", "count_n3", "count_rem",
]


def guardar_inventario(inventario, path=None):
    path = Path(path) if path else OUT_META / "inventory_records.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        for rid in sorted(inventario):
            writer.writerow(inventario[rid])
    log.info("Inventario: %s (%d registros)", path, len(inventario))


def leer_inventario(path=None):
    path = Path(path) if path else OUT_META / "inventory_records.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(INVENTORY_COLUMNS).issubset(rows[0]):
        raise ValueError("Inventario ausente u obsoleto: ejecutar --fase e0")
    return {row["record_id"]: row for row in rows}


def _contar_epocas(pares_por_particion, inventario):
    sizes = {}
    for part, pares in pares_por_particion.items():
        total = 0
        for psg, hyp, rid in pares:
            _, labels, starts, info = leer_sujeto(
                psg, hyp, devolver_inicios=True, solo_metadata=True
            )
            inventario[rid].update(info)
            total += len(labels)
            if len(starts) != len(labels):
                raise AssertionError(f"longitud inconsistente: {rid}")
        if not total:
            raise ValueError(f"Sin epocas validas para particion {part}")
        sizes[part] = total
        log.info("%s: %d epocas E0 previstas", part, total)
    return sizes


def _generar_tensores_e0(pares_por_particion, media, std, inventario,
                         subject_groups, record_groups, sizes):
    for part, pares in pares_por_particion.items():
        n = sizes[part]
        x_mm = np.lib.format.open_memmap(
            OUT_E0 / f"X_{part}.npy", mode="w+", dtype=np.float32,
            shape=(n, len(CANALES), N_MUESTRAS),
        )
        y = np.empty(n, dtype=np.int64)
        groups = np.empty(n, dtype=np.int64)
        records = np.empty(n, dtype=np.int64)
        starts_all = np.empty(n, dtype=np.int64)
        cursor = 0
        for psg, hyp, rid in pares:
            epocas, labels, starts, info = leer_sujeto(
                psg, hyp, devolver_inicios=True
            )
            n_reg = len(labels)
            if n_reg != int(inventario[rid]["epochs_valid_e0"]):
                raise RuntimeError(f"Cambio de numero de epocas entre pasadas: {rid}")
            if not n_reg:
                continue
            sid = rid[:5]
            x_mm[cursor:cursor + n_reg] = aplicar_normalizacion(epocas, media, std)
            y[cursor:cursor + n_reg] = labels
            groups[cursor:cursor + n_reg] = subject_groups[sid]
            records[cursor:cursor + n_reg] = record_groups[rid]
            starts_all[cursor:cursor + n_reg] = starts
            row = inventario[rid]
            for cls, name in CLASE_NOMBRES.items():
                row[f"count_{name.lower()}"] = int(np.count_nonzero(labels == cls))
            cursor += n_reg
            del epocas, labels, starts
            gc.collect()
        if cursor != n:
            raise AssertionError(f"Conteo E0 distinto para {part}: {cursor} != {n}")
        x_mm.flush()
        del x_mm
        np.save(OUT_E0 / f"y_{part}.npy", y)
        np.save(OUT_E0 / f"groups_{part}.npy", groups)
        np.save(OUT_E0 / f"record_ids_{part}.npy", records)
        np.save(OUT_E0 / f"epoch_starts_{part}.npy", starts_all)
        log.info("%s: E0 escrito, %d epocas", part, cursor)
        del y, groups, records, starts_all
        gc.collect()


def _centros_validos(starts, seq_len=SEQ_LEN):
    """Centros cuyo contexto tiene pasos EXACTOS de 3000 muestras."""
    starts = np.asarray(starts, dtype=np.int64)
    if seq_len % 2 != 1 or seq_len < 3:
        raise ValueError("seq_len debe ser impar y >= 3")
    if len(starts) < seq_len:
        return np.empty(0, dtype=np.int64)
    steps = np.diff(starts) == N_MUESTRAS
    # Para cada ventana de seq_len, todos sus seq_len-1 saltos deben ser reales.
    windows_ok = np.convolve(
        steps.astype(np.int64), np.ones(seq_len - 1, dtype=np.int64),
        mode="valid",
    ) == seq_len - 1
    return np.flatnonzero(windows_ok).astype(np.int64) + seq_len // 2


def construir_secuencias_registro(epocas, labels, seq_len=SEQ_LEN, starts=None):
    """Helper compatible con el test smoke; el pipeline exige starts reales.

    Si starts=None, se ASUME continuidad y no debe usarse para validar EDF.
    """
    n = len(labels)
    if n != len(epocas):
        raise ValueError("epocas y labels deben tener igual longitud")
    if starts is None:
        starts = np.arange(n, dtype=np.int64) * N_MUESTRAS
    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) != n:
        raise ValueError("starts debe tener una marca temporal por epoca")
    centers = _centros_validos(starts, seq_len=seq_len)
    if not len(centers):
        return None, None, None
    pad = seq_len // 2
    out = np.empty(
        (len(centers), seq_len, epocas.shape[1], epocas.shape[2]),
        dtype=np.float32,
    )
    for j, center in enumerate(centers):
        out[j] = epocas[center - pad:center + pad + 1]
    return out, np.asarray(labels)[centers].astype(np.int64), centers


def generar_tensores_seq(pares_por_particion, inventario, record_groups):
    """Genera ventanas solo dentro de cada registro con cronologia validada."""
    for part, pares in pares_por_particion.items():
        x_e0 = np.load(OUT_E0 / f"X_{part}.npy", mmap_mode="r")
        y_e0 = np.load(OUT_E0 / f"y_{part}.npy", mmap_mode="r")
        g_e0 = np.load(OUT_E0 / f"groups_{part}.npy", mmap_mode="r")
        r_e0 = np.load(OUT_E0 / f"record_ids_{part}.npy", mmap_mode="r")
        starts = np.load(OUT_E0 / f"epoch_starts_{part}.npy", mmap_mode="r")
        n_e0 = len(y_e0)
        if not all(len(a) == n_e0 for a in (x_e0, g_e0, r_e0, starts)):
            raise AssertionError(f"Dimensiones E0 inconsistentes: {part}")

        # Construir ventanas por registro; groups contiene PERSONAS, no noches.
        rec_info = []
        total = 0
        for _, _, rid in pares:
            rg = record_groups[rid]
            idx = np.flatnonzero(r_e0 == rg)
            n = len(idx)
            centers = _centros_validos(starts[idx])
            n_borde = min(n, SEQ_LEN - 1)
            n_gap = max(0, n - n_borde - len(centers))
            row = inventario[rid]
            if int(row["epochs_valid_e0"]) != n:
                raise ValueError(f"Inventario E0 desalineado: {rid}")
            row["epochs_excluded_border"] = n_borde
            row["epochs_excluded_discontinuity"] = n_gap
            row["epochs_valid_seq"] = int(len(centers))
            if n and not np.all(np.diff(idx) == 1):
                raise AssertionError(f"Registro fragmentado en E0: {rid}")
            if n and len(np.unique(g_e0[idx])) != 1:
                raise AssertionError(f"Una noche mezcla personas: {rid}")
            total += len(centers)
            rec_info.append((rid, idx, centers))

        if total <= 0:
            raise ValueError(f"Ninguna secuencia valida para {part}")
        mm = np.lib.format.open_memmap(
            OUT_SEQ / f"X_seq_{part}.npy", mode="w+", dtype=np.float32,
            shape=(total, SEQ_LEN, len(CANALES), N_MUESTRAS),
        )
        y_seq = np.empty(total, dtype=np.int64)
        g_seq = np.empty(total, dtype=np.int64)
        eligible = np.empty(total, dtype=np.int64)
        cursor = 0
        pad = SEQ_LEN // 2
        for rid, idx, centers in rec_info:
            for center in centers:
                e0_indices = idx[center - pad:center + pad + 1]
                mm[cursor] = x_e0[e0_indices]
                e0_center = int(idx[center])
                y_seq[cursor] = y_e0[e0_center]
                g_seq[cursor] = g_e0[e0_center]
                eligible[cursor] = e0_center
                cursor += 1
        if cursor != total:
            raise AssertionError(f"Conteo seq distinto para {part}")
        mm.flush()
        del mm
        if not np.array_equal(y_seq, y_e0[eligible]):
            raise AssertionError(f"Labels centrales mal alineadas: {part}")
        if not np.array_equal(g_seq, g_e0[eligible]):
            raise AssertionError(f"Personas no alineadas en secuencias: {part}")
        np.save(OUT_SEQ / f"y_seq_{part}.npy", y_seq)
        np.save(OUT_SEQ / f"groups_seq_{part}.npy", g_seq)
        np.save(OUT_SEQ / f"eligible_indices_{part}.npy", eligible)
        log.info("%s: seq=%d, E0=%d, sin saltos temporales", part, total, n_e0)
        del x_e0, y_e0, g_e0, r_e0, starts, y_seq, g_seq, eligible
        gc.collect()


def guardar_split_info(asignacion_sc, sujetos_st, media, std, pares_por_particion,
                       subject_groups, record_groups):
    sets = verificar_particion_por_persona(pares_por_particion)
    info = {
        "protocol_version": PROTOCOL_VERSION,
        "protocolo": "SC por persona; ST externo; ventanas adyacentes por registro",
        "split_seed": SPLIT_SEED,
        "split_proporciones": {"train": 0.64, "val": 0.16, "test": 0.20, "st": 1.0},
        "trim_wake_margin_min": TRIM_MARGIN_MIN,
        "canales": CANALES,
        "sample_rate_hz": FS,
        "epoch_duration_s": EPOCH_SEC,
        "n_muestras_epoca": N_MUESTRAS,
        "seq_len": SEQ_LEN,
        "norm_params_file": "norm_params.npz",
        "norm_media_por_canal": np.asarray(media).tolist(),
        "norm_std_por_canal": np.asarray(std).tolist(),
        "n_sujetos": {p: len(sets[p]) for p in PARTICIONES},
        "sujetos": {p: sorted(sets[p]) for p in PARTICIONES},
        "n_registros": {p: len(pares_por_particion[p]) for p in PARTICIONES},
        "registros": {p: sorted(r[2] for r in pares_por_particion[p]) for p in PARTICIONES},
        "mappings_file": "group_mappings.json",
        "conteos_e0": {}, "conteos_seq": {}, "exclusiones_borde": {},
    }
    for part in PARTICIONES:
        for target, dpath, name in (
            ("conteos_e0", OUT_E0, f"y_{part}.npy"),
            ("conteos_seq", OUT_SEQ, f"y_seq_{part}.npy"),
        ):
            path = dpath / name
            if path.exists():
                labels = np.load(path, mmap_mode="r")
                info[target][part] = {
                    "total": int(len(labels)),
                    "por_clase": {
                        CLASE_NOMBRES[c]: int(np.count_nonzero(labels == c))
                        for c in range(5)
                    },
                }
        n_e0 = info["conteos_e0"].get(part, {}).get("total", 0)
        n_seq = info["conteos_seq"].get(part, {}).get("total", 0)
        if n_e0 and n_seq:
            info["exclusiones_borde"][part] = {
                "epochs_e0": n_e0,
                "epochs_seq": n_seq,
                # Conserva el nombre historico pero incluye tambien las
                # exclusiones por discontinuidad temporal; se desglosan abajo.
                "excluded_border": n_e0 - n_seq,
            }

    inventario = leer_inventario()
    if info["conteos_seq"]:
        info["exclusiones_detalle"] = {
            p: {
                "border": sum(int(r["epochs_excluded_border"])
                              for r in inventario.values() if r["partition"] == p),
                "discontinuity": sum(int(r["epochs_excluded_discontinuity"])
                                     for r in inventario.values() if r["partition"] == p),
            }
            for p in PARTICIONES
        }
        for p in PARTICIONES:
            if p in info["conteos_seq"]:
                a = info["exclusiones_detalle"][p]
                assert (a["border"] + a["discontinuity"] ==
                        info["conteos_e0"][p]["total"] - info["conteos_seq"][p]["total"])
    (OUT_META / "split_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_META / "group_mappings.json").write_text(
        json.dumps({
            "protocol_version": PROTOCOL_VERSION,
            "subject_to_group": subject_groups,
            "record_to_group": record_groups,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("Metadatos guardados en %s", OUT_META)


def _verificar_fase_seq(pares_por_particion, subj_groups, rec_groups, asignacion):
    """Impide usar E0 obsoletos al generar secuencias por separado."""
    split_path = OUT_META / "split_info.json"
    map_path = OUT_META / "group_mappings.json"
    if not split_path.exists() or not map_path.exists():
        raise FileNotFoundError("Metadatos E0 faltantes: ejecutar --fase e0 primero")
    si = json.loads(split_path.read_text(encoding="utf-8"))
    mp = json.loads(map_path.read_text(encoding="utf-8"))
    if si.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Los E0 pertenecen a un protocolo anterior. Regenerar E0")
    if mp.get("subject_to_group") != subj_groups or mp.get("record_to_group") != rec_groups:
        raise ValueError("Inventario de sujetos/registros cambiado. Regenerar E0")
    for part in PARTICIONES:
        actual = sorted(r[2] for r in pares_por_particion[part])
        if si.get("registros", {}).get(part) != actual:
            raise ValueError(f"Particion de {part} cambio desde E0. Regenerar E0")
        for stem in ("X", "y", "groups", "record_ids", "epoch_starts"):
            if not (OUT_E0 / f"{stem}_{part}.npy").exists():
                raise FileNotFoundError(f"Falta E0: {stem}_{part}.npy")
    for sid, part in asignacion.items():
        if sid not in si["sujetos"][part]:
            raise ValueError("Split de personas no coincide con E0 anterior")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sleep-EDF: pipeline de tesis corregido")
    parser.add_argument("--fase", choices=["e0", "seq", "todo"], default="todo")
    # Admitir el layout de la tesis (data/raw) y el del microproyecto (data/).
    default_sc = RAW_SC if RAW_SC.is_dir() else ROOT / "data" / "sleep-cassette"
    default_st = RAW_ST if RAW_ST.is_dir() else ROOT / "data" / "sleep-telemetry"
    parser.add_argument("--raw-sc", type=Path, default=default_sc)
    parser.add_argument("--raw-st", type=Path, default=default_st)
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args(argv)

    global OUT_E0, OUT_SEQ, OUT_META
    out_root = args.out_root.resolve()
    OUT_E0, OUT_SEQ, OUT_META = (out_root / "e0", out_root / "seq", out_root / "meta")
    for path in (OUT_E0, OUT_SEQ, OUT_META):
        path.mkdir(parents=True, exist_ok=True)

    pares_sc = encontrar_pares(args.raw_sc)
    pares_st = encontrar_pares(args.raw_st)
    if not pares_sc or not pares_st:
        raise FileNotFoundError(
            f"Se necesitan PSG/Hypnogram SC y ST. SC={len(pares_sc)}, "
            f"ST={len(pares_st)}; comprueba --raw-sc y --raw-st"
        )
    if any(not rid.startswith("SC") for _, _, rid in pares_sc):
        raise ValueError("RAW_SC contiene registros que no son SC")
    if any(not rid.startswith("ST") for _, _, rid in pares_st):
        raise ValueError("RAW_ST contiene registros que no son ST")

    sujetos_sc = sorted({rid[:5] for _, _, rid in pares_sc})
    sujetos_st = sorted({rid[:5] for _, _, rid in pares_st})
    asignacion = split_sc_sujetos(sujetos_sc)
    pares_part = _pares_particiones(pares_sc, pares_st, asignacion)
    sets = verificar_particion_por_persona(pares_part)
    subj_groups, rec_groups = _mappings(pares_sc, pares_st)
    log.info("Personas por particion: %s", {p: len(sets[p]) for p in PARTICIONES})
    log.info("Registros por particion: %s", {p: len(pares_part[p]) for p in PARTICIONES})

    if args.fase in ("e0", "todo"):
        # Las secuencias anteriores NO pueden reutilizarse luego de regenerar E0.
        for f in OUT_SEQ.glob("*.npy"):
            f.unlink()
        inventario = {}
        for part in PARTICIONES:
            for psg, hyp, rid in pares_part[part]:
                inventario[rid] = _fila_inventario(
                    psg, hyp, rid, "ST" if part == "st" else "SC",
                    part, subj_groups, rec_groups,
                )
        sizes = _contar_epocas(pares_part, inventario)
        media, std = calcular_norm_params(pares_part["train"])
        np.savez(OUT_META / "norm_params.npz", media=media, std=std)
        log.info("Normalizacion solo SC-train: mean=%s std=%s", media, std)
        _generar_tensores_e0(
            pares_part, media, std, inventario, subj_groups, rec_groups, sizes
        )
        guardar_inventario(inventario)
        guardar_split_info(asignacion, sujetos_st, media, std, pares_part,
                           subj_groups, rec_groups)
    else:
        _verificar_fase_seq(pares_part, subj_groups, rec_groups, asignacion)
        inventario = leer_inventario()
        with np.load(OUT_META / "norm_params.npz") as npz:
            media, std = npz["media"], npz["std"]

    if args.fase in ("seq", "todo"):
        generar_tensores_seq(pares_part, inventario, rec_groups)
        guardar_inventario(inventario)
        guardar_split_info(asignacion, sujetos_st, media, std, pares_part,
                           subj_groups, rec_groups)

    log.info("Pipeline %s completado. Artefactos: %s", PROTOCOL_VERSION, out_root)


if __name__ == "__main__":
    main()
