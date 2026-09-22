from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


# configuracion

ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = ROOT / "data"
DIR_SC = BASE_DIR / "sleep-cassette"
DIR_ST = BASE_DIR / "sleep-telemetry"

OUTPUT_DIR = ROOT / "outputs" / "data"
OUTPUT_CSV = OUTPUT_DIR / "dataset_sleep_lightgbm.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_sleep_lightgbm.parquet"

EPOCH_DURATION = 30

CANALES_ML = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal"
]

EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 6,
    "Sleep stage R": 5
}

STAGE_NAMES = {
    1: "Wake",
    2: "N1",
    3: "N2",
    4: "N3",
    6: "N3",
    5: "REM"
}

BANDAS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}


# obtener ids

def id_registro(nombre_archivo):
    return nombre_archivo[:6]


def id_sujeto(nombre_archivo):
    return nombre_archivo[:5]


# emparejar archivos

def emparejar_archivos(carpeta):
    psg_files = sorted(
        carpeta.glob("*-PSG.edf")
    )

    hyp_files = sorted(
        carpeta.glob("*-Hypnogram.edf")
    )

    hyp_dict = {
        id_registro(archivo.name): archivo
        for archivo in hyp_files
    }

    pares = []

    for psg in psg_files:
        registro = id_registro(psg.name)
        sujeto = id_sujeto(psg.name)

        if registro in hyp_dict:
            pares.append({
                "id": registro,
                "subject": sujeto,
                "psg": psg,
                "hypnogram": hyp_dict[registro]
            })

        else:
            print(
                f"Sin Hypnogram: {psg.name}"
            )

    return pares


# extraer caracteristicas

def extraer_features_matriz(
    signals,
    fs
):
    signals = np.asarray(
        signals,
        dtype=np.float64
    )

    features = {
        "mean": np.mean(
            signals,
            axis=1
        ),
        "std": np.std(
            signals,
            axis=1
        ),
        "min": np.min(
            signals,
            axis=1
        ),
        "max": np.max(
            signals,
            axis=1
        ),
        "range": np.ptp(
            signals,
            axis=1
        ),
        "rms": np.sqrt(
            np.mean(
                signals ** 2,
                axis=1
            )
        )
    }

    nperseg = min(
        signals.shape[1],
        int(fs * 4)
    )

    freqs, psd = welch(
        signals,
        fs=fs,
        nperseg=nperseg,
        axis=1
    )

    for nombre, limites in BANDAS.items():
        fmin, fmax = limites

        mask = (
            (freqs >= fmin)
            & (freqs < fmax)
        )

        features[nombre] = np.trapezoid(
            psd[:, mask],
            freqs[mask],
            axis=1
        )

    return features


# procesar registro

def procesar_registro(
    psg_path,
    hyp_path,
    estudio,
    subject
):
    inicio = perf_counter()

    print(
        f"Procesando sujeto {subject}: "
        f"{psg_path.name}"
    )

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=False,
        verbose=False
    )

    canales_faltantes = [
        canal
        for canal in CANALES_ML
        if canal not in raw.ch_names
    ]

    if canales_faltantes:
        raw.close()

        raise ValueError(
            "Canales faltantes: "
            f"{canales_faltantes}"
        )

    raw.pick(CANALES_ML)
    raw.load_data()

    annotations = mne.read_annotations(
        hyp_path
    )

    raw.set_annotations(annotations)

    events, eventos_presentes = (
        mne.events_from_annotations(
            raw,
            event_id=EVENT_ID,
            chunk_duration=EPOCH_DURATION,
            verbose=False
        )
    )

    if len(events) == 0:
        raw.close()

        raise ValueError(
            "Sin eventos validos"
        )

    fs = raw.info["sfreq"]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=eventos_presentes,
        tmin=0,
        tmax=EPOCH_DURATION - 1 / fs,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
        verbose=False
    )

    if len(epochs) == 0:
        raw.close()

        raise ValueError(
            "No se generaron epochs validos"
        )

    datos = epochs.get_data()

    stages = [
        STAGE_NAMES[evento]
        for evento in epochs.events[:, 2]
    ]

    dataset = pd.DataFrame({
        "subject": subject,
        "study": estudio,
        "archivo": psg_path.name,
        "epoch": np.arange(
            len(epochs)
        ),
        "stage": stages
    })

    for indice, canal in enumerate(
        epochs.ch_names
    ):
        features = extraer_features_matriz(
            datos[:, indice, :],
            fs
        )

        nombre_canal = (
            canal
            .replace(" ", "_")
            .replace("-", "_")
        )

        for nombre_feature, valores in features.items():
            columna = (
                f"{nombre_canal}_"
                f"{nombre_feature}"
            )

            dataset[columna] = valores

    raw.close()

    duracion = perf_counter() - inicio

    print(
        f"Epochs: {len(dataset)} | "
        f"Tiempo: {duracion:.1f} segundos"
    )

    return dataset


# procesar estudio

def procesar_estudio(
    carpeta,
    nombre_estudio,
    limite=None
):
    pares = emparejar_archivos(
        carpeta
    )

    if limite is not None:
        pares = pares[:limite]

    print(
        f"\n{nombre_estudio}: "
        f"{len(pares)} registros"
    )

    datasets = []
    errores = []

    for indice, par in enumerate(
        pares,
        start=1
    ):
        print(
            f"\n[{indice}/{len(pares)}] "
            f"registro {par['id']} - "
            f"sujeto {par['subject']}"
        )

        try:
            dataset = procesar_registro(
                par["psg"],
                par["hypnogram"],
                nombre_estudio,
                par["subject"]
            )

            datasets.append(dataset)

        except Exception as error:
            mensaje = (
                f"ERROR {par['id']}: "
                f"{error}"
            )

            print(mensaje)
            errores.append(mensaje)

    if not datasets:
        return pd.DataFrame(), errores

    dataset_estudio = pd.concat(
        datasets,
        ignore_index=True
    )

    return dataset_estudio, errores


# validar dataset

def validar_dataset(dataset):
    columnas_control = {
        "subject",
        "study",
        "archivo",
        "epoch",
        "stage"
    }

    columnas_features = [
        columna
        for columna in dataset.columns
        if columna not in columnas_control
    ]

    valores_faltantes = (
        dataset[columnas_features]
        .isna()
        .sum()
        .sum()
    )

    valores_infinitos = np.isinf(
        dataset[columnas_features]
    ).sum().sum()

    print("\nValidacion:")

    print(
        f"Filas: {len(dataset)}"
    )

    print(
        f"Columnas: {dataset.shape[1]}"
    )

    print(
        f"Features: {len(columnas_features)}"
    )

    print(
        f"Valores faltantes: "
        f"{valores_faltantes}"
    )

    print(
        f"Valores infinitos: "
        f"{valores_infinitos}"
    )

    if len(columnas_features) != 30:
        raise ValueError(
            "Se esperaban 30 features, "
            f"pero se obtuvieron "
            f"{len(columnas_features)}"
        )

    if valores_faltantes > 0:
        raise ValueError(
            "El dataset contiene "
            "valores faltantes"
        )

    if valores_infinitos > 0:
        raise ValueError(
            "El dataset contiene "
            "valores infinitos"
        )


# guardar dataset

def guardar_dataset(dataset):
    dataset.to_csv(
        OUTPUT_CSV,
        index=False
    )

    dataset.to_parquet(
        OUTPUT_PARQUET,
        index=False
    )

    print(
        f"\nCSV guardado en: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Parquet guardado en: "
        f"{OUTPUT_PARQUET}"
    )


# guardar errores

def guardar_errores(errores):
    if not errores:
        return

    error_path = (
        OUTPUT_DIR
        / "errores_dataset_lightgbm.txt"
    )

    error_path.write_text(
        "\n".join(errores),
        encoding="utf-8"
    )

    print(
        f"\nRegistros con error: "
        f"{len(errores)}"
    )

    print(
        f"Detalle: {error_path}"
    )


# crear dataset

def main(limite=None):
    inicio_total = perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset_sc, errores_sc = procesar_estudio(
        DIR_SC,
        "sleep-cassette",
        limite
    )

    dataset_st, errores_st = procesar_estudio(
        DIR_ST,
        "sleep-telemetry",
        limite
    )

    datasets = [
        dataset
        for dataset in [
            dataset_sc,
            dataset_st
        ]
        if not dataset.empty
    ]

    if not datasets:
        raise RuntimeError(
            "No se genero ningun dataset"
        )

    dataset = pd.concat(
        datasets,
        ignore_index=True
    )

    validar_dataset(dataset)

    print(
        f"\nDimensiones: "
        f"{dataset.shape}"
    )

    print(
        f"Registros: "
        f"{dataset['archivo'].nunique()}"
    )

    print(
        f"Sujetos reales: "
        f"{dataset['subject'].nunique()}"
    )

    print("\nEtapas:")

    print(
        dataset["stage"].value_counts()
    )

    guardar_dataset(dataset)

    errores = (
        errores_sc
        + errores_st
    )

    guardar_errores(errores)

    duracion_total = (
        perf_counter()
        - inicio_total
    )

    print(
        f"\nTiempo total: "
        f"{duracion_total / 60:.1f} minutos"
    )


# argumentos

def obtener_argumentos():
    parser = ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cantidad maxima de registros "
            "por estudio"
        )
    )

    return parser.parse_args()


# ejecutar

if __name__ == "__main__":
    args = obtener_argumentos()

    main(
        limite=args.limit
    )