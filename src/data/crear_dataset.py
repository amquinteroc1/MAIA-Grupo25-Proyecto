from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from src.data.features import extraer_features


# configuracion

BASE_DIR = Path("data")
DIR_SC = BASE_DIR / "sleep-cassette"
DIR_ST = BASE_DIR / "sleep-telemetry"

OUTPUT_DIR = Path("outputs") / "data"
OUTPUT_CSV = OUTPUT_DIR / "dataset_sleep_ml.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_sleep_ml.parquet"

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


# extraer features vectorizadas sobre matriz de epocas

def extraer_features_matriz(signals, fs):
    signals = np.asarray(signals, dtype=np.float64)

    features = {
        "mean": np.mean(signals, axis=1),
        "std": np.std(signals, axis=1),
        "min": np.min(signals, axis=1),
        "max": np.max(signals, axis=1),
        "range": np.ptp(signals, axis=1),
        "rms": np.sqrt(np.mean(signals ** 2, axis=1))
    }

    nperseg = min(signals.shape[1], int(fs * 4))
    freqs, psd = welch(signals, fs=fs, nperseg=nperseg, axis=1)

    for nombre, (fmin, fmax) in BANDAS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        features[nombre] = np.trapezoid(psd[:, mask], freqs[mask], axis=1)

    return features


# obtener ids

def id_registro(nombre_archivo):
    return nombre_archivo[:6]


def id_sujeto(nombre_archivo):
    return nombre_archivo[:5]


# emparejar archivos

def emparejar_archivos(carpeta):
    psg_files = sorted(carpeta.glob("*-PSG.edf"))
    hyp_files = sorted(carpeta.glob("*-Hypnogram.edf"))

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
            print(f"Sin Hypnogram: {psg.name}")

    return pares


# procesar registro

def procesar_registro(
    psg_path,
    hyp_path,
    estudio,
    subject
):
    print(
        f"Procesando sujeto {subject}: "
        f"{psg_path.name}"
    )

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True,
        verbose=False
    )

    annotations = mne.read_annotations(
        hyp_path
    )

    raw.set_annotations(annotations)

    canales = [
        canal
        for canal in CANALES_ML
        if canal in raw.ch_names
    ]

    if not canales:
        raise ValueError(
            f"Sin canales validos en {psg_path.name}"
        )

    events, event_dict = mne.events_from_annotations(
        raw,
        event_id=EVENT_ID,
        chunk_duration=EPOCH_DURATION,
        verbose=False
    )

    if len(events) == 0:
        raise ValueError("Sin eventos validos")

    fs = raw.info["sfreq"]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=EPOCH_DURATION - 1 / fs,
        baseline=None,
        picks=canales,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    stages = [
        STAGE_NAMES.get(evento)
        for evento in epochs.events[:, 2]
    ]

    valid_indices = [
        i for i, st in enumerate(stages)
        if st is not None
    ]

    if not valid_indices:
        raw.close()
        return pd.DataFrame()

    datos = epochs.get_data()[valid_indices]
    raw.close()
    stages_valid = [stages[i] for i in valid_indices]

    df_res = pd.DataFrame({
        "subject": subject,
        "study": estudio,
        "archivo": psg_path.name,
        "epoch": valid_indices,
        "stage": stages_valid
    })

    for j, canal in enumerate(epochs.ch_names):
        feats = extraer_features_matriz(datos[:, j, :], fs)
        nombre_canal = canal.replace(" ", "_").replace("-", "_")
        for k, v in feats.items():
            df_res[f"{nombre_canal}_{k}"] = v

    return df_res


# procesar estudio

def procesar_estudio(
    carpeta,
    nombre_estudio
):
    pares = emparejar_archivos(carpeta)

    print(
        f"\n{nombre_estudio}: "
        f"{len(pares)} registros"
    )

    datasets = []

    for i, par in enumerate(
        pares,
        start=1
    ):
        print(
            f"[{i}/{len(pares)}] "
            f"registro {par['id']} - "
            f"sujeto {par['subject']}"
        )

        try:
            df = procesar_registro(
                par["psg"],
                par["hypnogram"],
                nombre_estudio,
                par["subject"]
            )

            datasets.append(df)

            print(f"Epochs: {len(df)}")

        except Exception as error:
            print(
                f"ERROR {par['id']}: "
                f"{error}"
            )

    if not datasets:
        return pd.DataFrame()

    return pd.concat(
        datasets,
        ignore_index=True
    )


# crear dataset

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if DIR_SC.exists():
        dataset_sc = procesar_estudio(
            DIR_SC,
            "sleep-cassette"
        )
    else:
        print(
            f"No existe: {DIR_SC.resolve()}"
        )
        dataset_sc = pd.DataFrame()

    if DIR_ST.exists():
        dataset_st = procesar_estudio(
            DIR_ST,
            "sleep-telemetry"
        )
    else:
        print(
            f"No existe: {DIR_ST.resolve()}"
        )
        dataset_st = pd.DataFrame()

    datasets = [
        dataset
        for dataset in [
            dataset_sc,
            dataset_st
        ]
        if not dataset.empty
    ]

    if not datasets:
        print(
            "No se genero ningun dataset"
        )

        print(
            f"Ruta esperada SC: "
            f"{DIR_SC.resolve()}"
        )

        print(
            f"Ruta esperada ST: "
            f"{DIR_ST.resolve()}"
        )

        return

    dataset = pd.concat(
        datasets,
        ignore_index=True
    )

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

    dataset.to_csv(
        OUTPUT_CSV,
        index=False
    )

    try:
        dataset.to_parquet(
            OUTPUT_PARQUET,
            index=False
        )

    except Exception as error:
        print(
            "No se pudo guardar "
            f"Parquet: {error}"
        )

    print(
        f"\nDataset CSV guardado en: "
        f"{OUTPUT_CSV.resolve()}"
    )

    if OUTPUT_PARQUET.exists():
        print(
            "Dataset Parquet guardado en: "
            f"{OUTPUT_PARQUET.resolve()}"
        )


# ejecutar

if __name__ == "__main__":
    main()