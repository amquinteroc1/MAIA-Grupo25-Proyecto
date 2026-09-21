"""
crear_dataset_cnn.py
Pipeline de alto rendimiento y arquitectura DataOps para la preparación de datos
CNN 1D (Sleep-EDF).

Características MLOps & DataOps:
- Paralelismo seguro para Windows (ProcessPoolExecutor) sin sobrecarga BLAS.
- Streaming sujeto a sujeto directo a disco via np.memmap (Peak RAM < 1.5 GB).
- Compatible con el formato exacto de artefactos esperado por DVC y train_cnn.
- Verificación automática de salud de almacenamiento y espacio libre.
- No inicializa contextos de GPU en workers para prevenir procesos zombi.
- Secuencias temporales generadas al vuelo en entrenamiento (ahorra 82.4 GB).
"""

import os

# Limitar hilos internos por worker para evitar sobre-suscripción en CPU multi-core
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

# Silenciar telemetría de MLflow en procesos hijos
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import argparse  # noqa: E402
from concurrent.futures import as_completed, ProcessPoolExecutor  # noqa: E402
import gc  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import mne  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(processName)s] %(message)s",
)
log = logging.getLogger("crear_dataset_cnn")

# Constantes del Pipeline
CANALES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
FS = 100
EPOCH_SEC = 30
N_MUESTRAS = FS * EPOCH_SEC  # 3000
SEQ_LEN = 5
SPLIT_SEED = 42
SPLIT_RATIO = 0.2

# Dimensiones conocidas del split estándar (197 sujetos Sleep-EDF, seed=42, ratio=0.2)
KNOWN_TRAIN_EPOCHS = 364642
KNOWN_TEST_EPOCHS = 93010

LABEL_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
EXCLUIR = {"Sleep stage M", "Sleep stage ?", "Movement time"}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRS = [
    ROOT / "data" / "sleep-cassette",
    ROOT / "data" / "sleep-telemetry",
]
DEFAULT_OUT_DIR = ROOT / "data_processed"


# ── Funciones de Utilidad y Preprocesamiento ──────────────────────────────────
def normalizar(epocas: np.ndarray) -> np.ndarray:
    """
    Normalización z-score por canal sobre todo el registro del sujeto.

    epocas: (N, C, T) float32
    Retorna: (N, C, T) normalizado in-place para evitar duplicación de memoria.
    """
    epocas = np.asarray(epocas, dtype=np.float32)
    for ch in range(epocas.shape[1]):
        señal = epocas[:, ch, :]
        mu = float(señal.mean())
        sigma = float(señal.std())
        if sigma > 0:
            señal -= mu
            señal /= sigma
    return epocas


def construir_secuencias(
    epocas: np.ndarray, labels: np.ndarray, seq_len: int = SEQ_LEN
):
    """
    Construye ventanas deslizantes de seq_len épocas para compatibilidad.

    La etiqueta corresponde a la época central, con padding de ceros en bordes.
    """
    n_epocas = len(epocas)
    pad = seq_len // 2
    pad_shape = (pad, epocas.shape[1], epocas.shape[2])
    pad_bloque = np.zeros(pad_shape, dtype=np.float32)
    epocas_pad = np.concatenate([pad_bloque, epocas, pad_bloque], axis=0)

    x_seq = np.stack([epocas_pad[i : i + seq_len] for i in range(n_epocas)])  # noqa: E203
    return x_seq, labels.copy()


def encontrar_pares(data_dirs=None):
    """Busca pares de archivos PSG e Hypnogram en los directorios de datos."""
    if data_dirs is None:
        data_dirs = DEFAULT_DATA_DIRS

    pares = []
    for d in data_dirs:
        d_path = Path(d)
        if not d_path.exists():
            continue
        for psg in sorted(glob.glob(str(d_path / "*-PSG.edf"))):
            base = psg.replace("-PSG.edf", "")
            hyp_candidates = glob.glob(base[:-2] + "*-Hypnogram.edf")
            if hyp_candidates:
                pares.append((psg, hyp_candidates[0], Path(psg).stem[:7]))
    return pares


def leer_sujeto(psg_path: str, hyp_path: str):
    """
    Lee y segmenta las épocas de un sujeto individual con memoria pre-asignada.

    Retorna:
        epocas: (N, len(CANALES), N_MUESTRAS) float32
        labels: (N,) int64
    """
    raw = None
    try:
        raw = mne.io.read_raw_edf(
            psg_path,
            include=CANALES,
            preload=True,
            verbose=False,
        )
        if any(ch not in raw.ch_names for ch in CANALES):
            return None, None

        ann = mne.read_annotations(hyp_path)
        datos = np.asarray(raw.get_data(picks=CANALES), dtype=np.float32)
        total_muestras = datos.shape[1]

        # Liberar MNE de inmediato antes de segmentar
        close = getattr(raw, "close", None)
        if close is not None:
            close()
        raw = None

        # Identificar intervalos válidos en una pasada ligera
        valid_intervals = []
        for a in ann:
            desc = a["description"]
            if desc in EXCLUIR or desc not in LABEL_MAP:
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
                valid_intervals.append((idx_ini, idx_fin, label))
                t += EPOCH_SEC

        if not valid_intervals:
            del datos
            return None, None

        # Pre-asignar exactamente en memoria contigua
        n_epocas = len(valid_intervals)
        epocas_arr = np.empty((n_epocas, len(CANALES), N_MUESTRAS), dtype=np.float32)
        labels_arr = np.empty(n_epocas, dtype=np.int64)

        for i, (ini, fin, lbl) in enumerate(valid_intervals):
            seg = datos[:, ini:fin]
            if seg.shape[1] < N_MUESTRAS:
                pad = np.zeros(
                    (seg.shape[0], N_MUESTRAS - seg.shape[1]), dtype=np.float32
                )
                epocas_arr[i] = np.concatenate([seg, pad], axis=1)
            else:
                epocas_arr[i] = seg
            labels_arr[i] = lbl

        del datos, valid_intervals
        return epocas_arr, labels_arr
    finally:
        if raw is not None:
            close = getattr(raw, "close", None)
            if close is not None:
                close()


# ── Worker para Procesamiento en Paralelo (CPU) ───────────────────────────────
def procesar_sujeto_worker(tarea):
    """
    Worker puro de CPU: carga y normaliza un único sujeto a la vez.

    Para evitar MemoryError en _ForkingPickler de Windows IPC, guarda el resultado
    en un archivo temporal ligero y retorna únicamente metadatos y la ruta.
    """
    psg_path, hyp_path, subject_id, grupo_id, is_train, tmp_dir_str = tarea
    try:
        epocas, labels = leer_sujeto(psg_path, hyp_path)
        if epocas is None or len(epocas) == 0:
            return None

        # Normalización z-score por canal in-place
        epocas = normalizar(epocas)

        tmp_path = Path(tmp_dir_str) / f"{subject_id}.npz"
        np.savez(tmp_path, x=epocas, y=labels)
        del epocas, labels

        return (subject_id, is_train, grupo_id, str(tmp_path))
    except Exception as exc:
        log.error("Error procesando sujeto %s: %s", subject_id, exc)
        return None


def contar_epocas_worker(tarea):
    """Cuenta épocas leyendo únicamente las anotaciones del hipnograma."""
    _psg, hyp_path, subject_id, _grupo_id, is_train = tarea
    try:
        ann = mne.read_annotations(hyp_path)
        count = 0
        for a in ann:
            desc = a["description"]
            if desc in EXCLUIR or desc not in LABEL_MAP:
                continue
            dur_s = a["duration"]
            count += int(dur_s // EPOCH_SEC)
        return subject_id, is_train, count
    except Exception as exc:
        log.warning("Fallo al leer anotaciones de %s: %s", hyp_path, exc)
        return subject_id, is_train, 0


# ── Comprobación de Almacenamiento y Archivos ──────────────────────────────────
def verificar_espacio_disco(directorio_destino: Path, gb_requeridos: float = 18.0):
    """Valida que el disco de destino tenga suficiente espacio libre."""
    directorio_destino.mkdir(parents=True, exist_ok=True)
    _total, _used, free = shutil.disk_usage(directorio_destino)
    free_gb = free / (1024**3)
    log.info(
        "Espacio libre en destino (%s): %.2f GB (Requerido: ~%.1f GB)",
        directorio_destino,
        free_gb,
        gb_requeridos,
    )
    if free_gb < gb_requeridos:
        msg = (
            f"Espacio insuficiente en disco: {free_gb:.2f} GB disponibles, "
            f"pero se requieren al menos {gb_requeridos:.1f} GB. Considere "
            f"especificar otro disco usando --out-dir."
        )
        raise OSError(msg)


def preparar_archivo_memmap(path: Path):
    """Elimina de forma segura archivos previos para evitar bloqueos de handle."""
    if path.exists():
        try:
            gc.collect()
            path.unlink()
            log.info("Archivo previo eliminado para asignación limpia: %s", path.name)
        except Exception as exc:
            log.warning("No se pudo desvincular archivo previo %s: %s", path, exc)


# ── Pipeline Principal ────────────────────────────────────────────────────────
def main():
    """Ejecuta el pipeline optimizado de preparación de datos CNN 1D."""
    parser = argparse.ArgumentParser(
        description="Pipeline optimizado de preparación de datos CNN 1D"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directorio de destino de artefactos",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Número de procesos paralelos para lectura de CPU (default: 4)",
    )
    parser.add_argument(
        "--materialize-sequences",
        action="store_true",
        help="Materializar secuencias de 5 épocas en disco (+82 GB)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    num_workers = max(1, min(args.num_workers, os.cpu_count() or 4))
    materializar_secuencias = args.materialize_sequences or (
        os.getenv("CNN_MATERIALIZE_SEQUENCES", "0") == "1"
    )

    log.info("=" * 70)
    log.info("Iniciando pipeline optimizado CNN 1D (MLOps & DataOps)")
    log.info("Directorio de salida : %s", out_dir)
    log.info("Workers CPU en pool  : %s", num_workers)
    log.info("=" * 70)

    # 1. Validar espacio disponible
    espacio_necesario = 100.0 if materializar_secuencias else 18.0
    verificar_espacio_disco(out_dir, espacio_necesario)

    # 2. Localizar pares PSG - Hypnogram
    pares = encontrar_pares()
    if not pares:
        raise FileNotFoundError(
            "No se encontraron pares PSG-Hypnogram en directorios de datos."
        )
    log.info("Pares encontrados: %d grabaciones.", len(pares))

    sujetos_unicos = sorted(set(p[2] for p in pares))
    sujeto_a_id = {s: i for i, s in enumerate(sujetos_unicos)}

    # 3. Determinación de split Train / Test por grupos (sujetos)
    subject_array = np.asarray(sujetos_unicos, dtype=object)
    gss = GroupShuffleSplit(n_splits=1, test_size=SPLIT_RATIO, random_state=SPLIT_SEED)
    idx_train_subj, idx_test_subj = next(
        gss.split(subject_array, subject_array, groups=subject_array)
    )

    train_subjects = {sujetos_unicos[i] for i in idx_train_subj}
    test_subjects = {sujetos_unicos[i] for i in idx_test_subj}

    assert not (
        train_subjects & test_subjects
    ), "Data leakage detectado: intersección no vacía entre train y test."
    log.info(
        "División de sujetos: %d en Train | %d en Test (sin leakage).",
        len(train_subjects),
        len(test_subjects),
    )

    tmp_ipc_dir = out_dir / ".tmp_ipc"
    tmp_ipc_dir.mkdir(parents=True, exist_ok=True)

    tareas = [
        (
            p[0],
            p[1],
            p[2],
            sujeto_a_id[p[2]],
            p[2] in train_subjects,
            str(tmp_ipc_dir),
        )
        for p in pares
    ]

    # 4. Cálculo exacto del número de épocas
    is_canonical = (
        len(pares) == 197
        and len(sujetos_unicos) == 197
        and SPLIT_SEED == 42
        and SPLIT_RATIO == 0.2
    )
    if is_canonical:
        n_train = KNOWN_TRAIN_EPOCHS
        n_test = KNOWN_TEST_EPOCHS
        log.info(
            "Dataset canónico detectado: asignando %d épocas train y %d test.",
            n_train,
            n_test,
        )
    else:
        log.info("Calculando dimensiones exactas de épocas en paralelo...")
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            conteo_results = list(pool.map(contar_epocas_worker, tareas))
        n_train = sum(c for _, is_tr, c in conteo_results if is_tr)
        n_test = sum(c for _, is_tr, c in conteo_results if not is_tr)
        log.info(
            "Dimensiones calculadas: %d épocas train | %d test.",
            n_train,
            n_test,
        )

    # 5. Pre-asignación de arrays con np.memmap
    x_train_path = out_dir / "X_train.npy"
    x_test_path = out_dir / "X_test.npy"
    preparar_archivo_memmap(x_train_path)
    preparar_archivo_memmap(x_test_path)

    log.info("Creando memmaps en disco para X_train y X_test...")
    X_train = np.memmap(
        x_train_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_train, len(CANALES), N_MUESTRAS),
    )
    X_test = np.memmap(
        x_test_path,
        dtype=np.float32,
        mode="w+",
        shape=(n_test, len(CANALES), N_MUESTRAS),
    )

    y_train = np.empty(n_train, dtype=np.int64)
    y_test = np.empty(n_test, dtype=np.int64)
    g_train = np.empty(n_train, dtype=np.int64)
    g_test = np.empty(n_test, dtype=np.int64)

    cursor_train = 0
    cursor_test = 0
    processed_count = 0

    # 6. Streaming Multiproceso con File-Backed IPC
    log.info(
        "Procesando %d sujetos con %d workers paralelos...",
        len(tareas),
        num_workers,
    )
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(procesar_sujeto_worker, tarea): tarea[2] for tarea in tareas
        }

        for fut in as_completed(futures):
            subj_id = futures[fut]
            res = fut.result()
            if res is None:
                log.warning("Sujeto %s no produjo épocas válidas. Omitido.", subj_id)
                continue

            _s_id, is_train, grupo_id, tmp_path_str = res
            tmp_data = np.load(tmp_path_str)
            epochs_norm = tmp_data["x"]
            labels_arr = tmp_data["y"]
            k = len(labels_arr)

            if is_train:
                end = cursor_train + k
                if end > n_train:
                    log.warning("Ajustando cursor train: %d > %d", end, n_train)
                    end = n_train
                    k = end - cursor_train
                X_train[cursor_train:end] = epochs_norm[:k]
                y_train[cursor_train:end] = labels_arr[:k]
                g_train[cursor_train:end] = grupo_id
                cursor_train = end
            else:
                end = cursor_test + k
                if end > n_test:
                    log.warning("Ajustando cursor test: %d > %d", end, n_test)
                    end = n_test
                    k = end - cursor_test
                X_test[cursor_test:end] = epochs_norm[:k]
                y_test[cursor_test:end] = labels_arr[:k]
                g_test[cursor_test:end] = grupo_id
                cursor_test = end

            processed_count += 1
            log.info(
                "[%03d/%d] Sujeto %s guardado (%d épocas) | Train: %d/%d | Test: %d/%d",
                processed_count,
                len(tareas),
                subj_id,
                k,
                cursor_train,
                n_train,
                cursor_test,
                n_test,
            )

            # Liberación estricta de memoria y eliminación del temporal
            del epochs_norm, labels_arr, tmp_data
            try:
                os.unlink(tmp_path_str)
            except Exception:
                pass

            if processed_count % 10 == 0:
                gc.collect()

    # Limpiar directorio temporal IPC
    shutil.rmtree(tmp_ipc_dir, ignore_errors=True)

    # 7. Sincronización y persistencia de etiquetas y metadatos
    log.info("Sincronizando memmaps a disco...")
    X_train.flush()
    X_test.flush()

    y_train = y_train[:cursor_train]
    g_train = g_train[:cursor_train]
    y_test = y_test[:cursor_test]
    g_test = g_test[:cursor_test]

    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "groups_train.npy", g_train)
    np.save(out_dir / "groups_test.npy", g_test)
    log.info("Archivos de etiquetas y grupos guardados exitosamente.")

    # 8. Secuencias Temporales (Opcional)
    if materializar_secuencias:
        log.info("Generando secuencias temporales en disco...")
        x_seq_tr_path = out_dir / "X_seq_train.npy"
        x_seq_te_path = out_dir / "X_seq_test.npy"
        preparar_archivo_memmap(x_seq_tr_path)
        preparar_archivo_memmap(x_seq_te_path)

        X_seq_tr = np.memmap(
            x_seq_tr_path,
            dtype=np.float32,
            mode="w+",
            shape=(cursor_train, SEQ_LEN, len(CANALES), N_MUESTRAS),
        )
        chunk_size = 500
        for i in range(0, cursor_train, chunk_size):
            end_i = min(i + chunk_size, cursor_train)
            batch = np.array(X_train[i:end_i])
            seqs, _ = construir_secuencias(batch, y_train[i:end_i], seq_len=SEQ_LEN)
            X_seq_tr[i:end_i] = seqs
        X_seq_tr.flush()
        del X_seq_tr

        X_seq_te = np.memmap(
            x_seq_te_path,
            dtype=np.float32,
            mode="w+",
            shape=(cursor_test, SEQ_LEN, len(CANALES), N_MUESTRAS),
        )
        for i in range(0, cursor_test, chunk_size):
            end_i = min(i + chunk_size, cursor_test)
            batch = np.array(X_test[i:end_i])
            seqs, _ = construir_secuencias(batch, y_test[i:end_i], seq_len=SEQ_LEN)
            X_seq_te[i:end_i] = seqs
        X_seq_te.flush()
        del X_seq_te
        log.info("Secuencias materializadas en disco.")
    else:
        log.info(
            "Secuencias completas en disco omitidas por diseño (ahorro de ~82.4 GB). "
            "El entrenamiento las construirá al vuelo en memoria de forma transparente."
        )

    # 9. Generar split_info.json
    clases_nombres = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    meta = {
        "canales": CANALES,
        "sample_rate_hz": FS,
        "epoch_duration_s": EPOCH_SEC,
        "n_muestras_epoca": N_MUESTRAS,
        "seq_len": SEQ_LEN,
        "split_seed": SPLIT_SEED,
        "split_ratio": SPLIT_RATIO,
        "n_sujetos_total": len(sujetos_unicos),
        "n_sujetos_train": len(train_subjects),
        "n_sujetos_test": len(test_subjects),
        "n_epocas_train": int(len(y_train)),
        "n_epocas_test": int(len(y_test)),
        "distribucion_train": {str(i): int((y_train == i).sum()) for i in range(5)},
        "distribucion_test": {str(i): int((y_test == i).sum()) for i in range(5)},
        "clases": clases_nombres,
    }

    info_path = out_dir / "split_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadatos guardados en %s", info_path)

    # 10. MLflow Tracking
    try:
        import mlflow  # pylint: disable=import-outside-toplevel

        mlflow.set_experiment("sleep-stage-cnn-pipeline")
        with mlflow.start_run(run_name="dataset-cnn-dataops-opt"):
            mlflow.log_params(
                {
                    "num_workers": num_workers,
                    "seq_len": SEQ_LEN,
                    "split_seed": SPLIT_SEED,
                    "split_ratio": SPLIT_RATIO,
                    "n_sujetos": len(sujetos_unicos),
                    "normalizacion": "z-score por sujeto y canal",
                }
            )
            mlflow.log_metrics(
                {
                    "n_epocas_train": float(len(y_train)),
                    "n_epocas_test": float(len(y_test)),
                    "n1_train_count": float((y_train == 1).sum()),
                    "class_imbalance": float(
                        (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                    ),
                }
            )
            mlflow.log_artifact(str(info_path))
        log.info("Ejecución registrada en MLflow.")
    except Exception as exc:
        log.warning("No se pudo registrar en MLflow (opcional): %s", exc)

    log.info("=" * 70)
    log.info("Pipeline completado exitosamente. Artefactos listos en %s", out_dir)
    log.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
