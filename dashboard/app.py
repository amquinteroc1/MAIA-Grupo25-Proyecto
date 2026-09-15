from pathlib import Path
import sys
import tempfile

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st


# configuracion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import preparar_features, predecir


MODELOS = {
    "LightGBM": (
        ROOT
        / "models"
        / "lightgbm_sleep.joblib"
    )
}

CANALES = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal"
]


st.set_page_config(
    page_title="Monitor del Sueño AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# estilos

st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
        color: #172033;
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #dfe4ea;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] * {
        color: #172033 !important;
    }

    [data-testid="stSidebar"] h1 {
        color: #123b63 !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #244b70 !important;
    }

    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border-radius: 12px;
    }

    .stButton > button {
        background-color: #ffffff;
        color: #123b63 !important;
        border: 1px solid #123b63;
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #edf5fc;
        color: #123b63 !important;
        border-color: #123b63;
    }

    .pred-card {
        padding: 28px;
        border-radius: 18px;
        background-color: #f7f9fc;
        text-align: center;
        border: 1px solid #dfe4ea;
        margin-bottom: 18px;
    }

    .stage {
        font-size: 56px;
        font-weight: 700;
        color: #123b63;
        line-height: 1;
    }

    .conf {
        font-size: 26px;
        color: #536b82;
        margin-top: 10px;
    }

    .model-card {
        padding: 12px;
        border-radius: 10px;
        background-color: #edf5fc;
        border: 1px solid #d8e8f5;
        color: #123b63;
        margin-bottom: 14px;
    }

    h1,
    h2,
    h3 {
        color: #172033;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# cargar modelo

@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)


# cargar edf

@st.cache_resource
def cargar_edf_bytes(file_bytes):
    ruta_temporal = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".edf"
        ) as archivo_temporal:
            archivo_temporal.write(
                file_bytes
            )

            ruta_temporal = Path(
                archivo_temporal.name
            )

        raw = mne.io.read_raw_edf(
            ruta_temporal,
            preload=True,
            verbose=False
        )

        return raw

    finally:
        if (
            ruta_temporal is not None
            and ruta_temporal.exists()
        ):
            try:
                ruta_temporal.unlink()

            except PermissionError:
                pass


# validar edf

def validar_edf(raw):
    faltantes = [
        canal
        for canal in CANALES
        if canal not in raw.ch_names
    ]

    if faltantes:
        raise ValueError(
            "El EDF no contiene los canales "
            f"requeridos: {faltantes}"
        )


# validar features

def validar_features(
    modelo,
    X
):
    if X.shape[1] != 30:
        raise ValueError(
            "El modelo requiere 30 features, "
            f"pero se generaron {X.shape[1]}"
        )

    if hasattr(
        modelo,
        "feature_name_"
    ):
        esperadas = list(
            modelo.feature_name_
        )

        actuales = list(
            X.columns
        )

        if actuales != esperadas:
            raise ValueError(
                "El orden de las features "
                "no coincide con el entrenamiento"
            )


# graficar señal

def grafico_signal(
    signal,
    fs,
    titulo
):
    tiempo = (
        np.arange(len(signal))
        / fs
    )

    signal_uv = signal * 1e6

    fig, ax = plt.subplots(
        figsize=(10, 2)
    )

    ax.plot(
        tiempo,
        signal_uv,
        linewidth=0.8,
        color="#123b63"
    )

    ax.set_title(titulo)
    ax.set_xlabel("Segundos")
    ax.set_ylabel("µV")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig


# sidebar

with st.sidebar:
    st.title(
        "Monitor del Sueño AI"
    )

    st.markdown(
        "### Señales"
    )

    mostrar_fpz = st.checkbox(
        "EEG Fpz-Cz",
        value=True
    )

    mostrar_pz = st.checkbox(
        "EEG Pz-Oz",
        value=True
    )

    mostrar_eog = st.checkbox(
        "EOG horizontal",
        value=True
    )

    archivo = st.file_uploader(
        "Cargar archivo PSG EDF",
        type=["edf"]
    )

    st.markdown(
        "### Modelo"
    )

    modelo_nombre = st.radio(
        "Seleccionar modelo",
        list(MODELOS.keys()),
        index=0,
        label_visibility="collapsed"
    )

    model_path = MODELOS[
        modelo_nombre
    ]

    st.markdown(
        f"""
        <div class="model-card">
            Modelo activo: <b>{modelo_nombre}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    analizar = st.button(
        "Analizar",
        use_container_width=True,
        type="primary"
    )


# layout

col_centro, col_derecha = (
    st.columns(
        [3.2, 1.1],
        gap="large"
    )
)


# contenido

if archivo is None:
    with col_centro:
        st.info(
            "Carga un archivo PSG EDF "
            "para comenzar."
        )

else:
    try:
        raw = cargar_edf_bytes(
            archivo.getvalue()
        )

        validar_edf(raw)

        fs = raw.info["sfreq"]
        duracion = raw.times[-1]

        max_ventanas = int(
            duracion // 30
        )

        with col_centro:
            st.subheader(
                "Señales"
            )

            st.caption(
                f"Archivo: {archivo.name} · "
                f"Frecuencia: {fs:.0f} Hz · "
                f"Duración: {duracion / 3600:.2f} h"
            )

            ventana = st.slider(
                "Ventana de 30 segundos",
                min_value=0,
                max_value=max(
                    0,
                    max_ventanas - 1
                ),
                value=0
            )

            inicio = ventana * 30
            start = int(inicio * fs)
            stop = int(
                (inicio + 30) * fs
            )

            canales_mostrar = []

            if mostrar_fpz:
                canales_mostrar.append(
                    "EEG Fpz-Cz"
                )

            if mostrar_pz:
                canales_mostrar.append(
                    "EEG Pz-Oz"
                )

            if mostrar_eog:
                canales_mostrar.append(
                    "EOG horizontal"
                )

            if not canales_mostrar:
                st.warning(
                    "Selecciona al menos "
                    "una señal para visualizar."
                )

            for canal in canales_mostrar:
                signal = raw.get_data(
                    picks=[canal],
                    start=start,
                    stop=stop
                )[0]

                fig = grafico_signal(
                    signal,
                    fs,
                    canal
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

        with col_derecha:
            st.subheader(
                "Predicción"
            )

            if analizar:
                if not model_path.exists():
                    st.error(
                        "No se encontró el modelo: "
                        f"{model_path}"
                    )

                    st.info(
                        "Ejecuta `dvc pull` para "
                        "recuperar el modelo."
                    )

                else:
                    try:
                        modelo = cargar_modelo(
                            model_path
                        )

                        X = preparar_features(
                            raw,
                            inicio
                        )

                        validar_features(
                            modelo,
                            X
                        )

                        etapa, probs = predecir(
                            modelo,
                            X
                        )

                        confianza = probs.get(
                            etapa,
                            0.0
                        )

                        st.markdown(
                            f"""
                            <div class="pred-card">
                                <div class="stage">
                                    {etapa}
                                </div>
                                <div class="conf">
                                    {confianza:.0%}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            "### Probabilidades"
                        )

                        orden = [
                            "Wake",
                            "N1",
                            "N2",
                            "N3",
                            "REM"
                        ]

                        for clase in orden:
                            prob = probs.get(
                                clase,
                                0.0
                            )

                            st.write(
                                f"{clase}: "
                                f"{prob:.1%}"
                            )

                            st.progress(
                                min(
                                    max(
                                        float(prob),
                                        0.0
                                    ),
                                    1.0
                                )
                            )

                    except Exception as error:
                        st.error(
                            "Error durante la "
                            f"predicción: {error}"
                        )

            else:
                st.info(
                    "Selecciona una ventana "
                    "y pulsa Analizar."
                )

    except Exception as error:
        with col_centro:
            st.error(
                "No fue posible leer "
                f"el EDF: {error}"
            )