from pathlib import Path
import mne
import pandas as pd

from src.data.features import extraer_features


# configuracion

BASE_DIR = Path("data")
DIR_SC = BASE_DIR / "sleep-cassette"
DIR_ST = BASE_DIR / "sleep-telemetry"

OUTPUT_DIR = Path("outputs") / "data"
OUTPUT_CSV = OUTPUT_DIR / "dataset_sleep_ml.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "dataset_sleep_ml.parquet"

EPOCH_DURATION = 30

CANALES_ML = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5
}

STAGE_NAMES = {1: "Wake", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}


# obtener id

def id_registro(nombre_archivo):
    return nombre_archivo[:6]


# emparejar archivos

def emparejar_archivos(carpeta):
    psg_files = sorted(carpeta.glob("*-PSG.edf"))
    hyp_files = sorted(carpeta.glob("*-Hypnogram.edf"))
    hyp_dict = {id_registro(f.name): f for f in hyp_files}
    pares = []

    for psg in psg_files:
        registro = id_registro(psg.name)

        if registro in hyp_dict:
            pares.append({"id": registro, "psg": psg, "hypnogram": hyp_dict[registro]})
        else:
            print(f"Sin Hypnogram: {psg.name}")

    return pares


# procesar registro

def procesar_registro(psg_path, hyp_path, estudio, subject):
    print(f"Procesando {subject}: {psg_path.name}")

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations)

    canales = [c for c in CANALES_ML if c in raw.ch_names]

    if not canales:
        raise ValueError(f"Sin canales validos en {psg_path.name}")

    events, _ = mne.events_from_annotations(
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
        event_id=EVENT_ID,
        tmin=0,
        tmax=EPOCH_DURATION - 1 / fs,
        baseline=None,
        picks=canales,
        preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    filas = []

    for i in range(len(epochs)):
        stage = STAGE_NAMES.get(epochs.events[i, 2])

        if stage is None:
            continue

        fila = {
            "subject": subject,
            "study": estudio,
            "archivo": psg_path.name,
            "epoch": i,
            "stage": stage
        }

        datos_epoch = epochs[i].get_data()

        for j, canal in enumerate(epochs.ch_names):
            features = extraer_features(datos_epoch[0, j, :], fs)
            nombre_canal = canal.replace(" ", "_").replace("-", "_")

            for nombre_feature, valor in features.items():
                fila[f"{nombre_canal}_{nombre_feature}"] = valor

        filas.append(fila)

    return pd.DataFrame(filas)


# procesar estudio

def procesar_estudio(carpeta, nombre_estudio):
    pares = emparejar_archivos(carpeta)
    print(f"\n{nombre_estudio}: {len(pares)} registros")

    datasets = []

    for i, par in enumerate(pares, start=1):
        print(f"[{i}/{len(pares)}]")

        try:
            df = procesar_registro(par["psg"], par["hypnogram"], nombre_estudio, par["id"])
            datasets.append(df)
            print(f"Epochs: {len(df)}")
        except Exception as e:
            print(f"ERROR {par['id']}: {e}")

    return pd.concat(datasets, ignore_index=True) if datasets else pd.DataFrame()


# crear dataset

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_sc = procesar_estudio(DIR_SC, "sleep-cassette") if DIR_SC.exists() else pd.DataFrame()
    dataset_st = procesar_estudio(DIR_ST, "sleep-telemetry") if DIR_ST.exists() else pd.DataFrame()

    datasets = [df for df in [dataset_sc, dataset_st] if not df.empty]

    if not datasets:
        print("No se genero ningun dataset")
        print(f"Ruta esperada SC: {DIR_SC.resolve()}")
        print(f"Ruta esperada ST: {DIR_ST.resolve()}")
        return

    dataset = pd.concat(datasets, ignore_index=True)

    print(f"\nDimensiones: {dataset.shape}")
    print(f"Sujetos: {dataset['subject'].nunique()}")
    print("\nEtapas:")
    print(dataset["stage"].value_counts())

    dataset.to_csv(OUTPUT_CSV, index=False)

    try:
        dataset.to_parquet(OUTPUT_PARQUET, index=False)
    except Exception as e:
        print(f"No se pudo guardar Parquet: {e}")

    print(f"\nDataset guardado en: {OUTPUT_CSV}")


# ejecutar

if __name__ == "__main__":
    main()