from pathlib import Path
import sys
import tempfile

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import streamlit as st

# ── Configuración y Rutas ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import (
    cargar_modelo,
    preparar_features,
    predecir,
    cargar_modelo_cnn,
    preparar_tensor_cnn,
    predecir_cnn,
)


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

SVM_MODEL_PATH = ROOT / "models" / "svm_sleep.joblib"
CNN_MODEL_PATH = ROOT / "experiments" / "cnn1d" / "best_cnn1d.pt"

st.set_page_config(
    page_title="Monitor del Sueño AI",
    layout="wide",
    initial_sidebar_state="expanded",
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
# ── Estilos Visuales ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #172033;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #f7f9fc;
    border-right: 1px solid #dfe4ea;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
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
    background-color: #123b63 !important;
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1rem;
}
.stButton > button:hover {
    background-color: #1b4f84 !important;
    color: #ffffff !important;
}
.pred-card {
    padding: 24px;
    border-radius: 16px;
    background-color: #f0f4f9;
    text-align: center;
    border: 1px solid #d0dbe7;
    margin-bottom: 18px;
}
.stage {
    font-size: 52px;
    font-weight: 800;
    color: #123b63;
    line-height: 1.1;
}
.stage-desc {
    font-size: 15px;
    font-weight: 600;
    color: #3b5a7a;
    margin-top: 4px;
}
.conf {
    font-size: 24px;
    color: #244b70;
    font-weight: 700;
    margin-top: 8px;
}
.model-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    background-color: #e2eaf3;
    color: #123b63;
    margin-bottom: 8px;
}
h1, h2, h3 {
    color: #172033;
}
</style>
""", unsafe_allow_html=True)

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
# ── Caché de Modelos ──────────────────────────────────────────────────────────
@st.cache_resource
def get_svm_model():
    if not SVM_MODEL_PATH.exists():
        return None
    return cargar_modelo(SVM_MODEL_PATH)

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
@st.cache_resource
def get_cnn_model():
    if not CNN_MODEL_PATH.exists():
        return None
    return cargar_modelo_cnn(CNN_MODEL_PATH)


# ── Caché de Lectura EDF ──────────────────────────────────────────────────────
@st.cache_resource
def cargar_edf_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file_bytes)
        ruta = tmp.name
    return mne.io.read_raw_edf(ruta, preload=True, verbose=False)

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

# ── Graficador de Señal ───────────────────────────────────────────────────────
def grafico_signal(signal, fs, titulo):
    tiempo = np.arange(len(signal)) / fs
    signal_uv = signal * 1e6

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(tiempo, signal_uv, linewidth=0.8, color="#123b63")
    ax.set_title(titulo, fontsize=11, fontweight="600")
    ax.set_xlabel("Segundos", fontsize=9)
    ax.set_ylabel("µV", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    st.markdown("### Señales a Visualizar")
    mostrar_fpz = st.checkbox("EEG Fpz-Cz", value=True)
    mostrar_pz = st.checkbox("EEG Pz-Oz", value=True)
    mostrar_eog = st.checkbox("EOG horizontal", value=True)

    st.markdown("### Archivo PSG")
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
        "Cargar archivo EDF",
        type=["edf"],
        help="Cargue un archivo *-PSG.edf con registros de polisomnografía.",
    )

    st.markdown("### Modelo Clasificador")

    opciones_modelos = []
    if CNN_MODEL_PATH.exists():
        opciones_modelos.append("CNN 1D (Deep Learning)")
    if SVM_MODEL_PATH.exists():
        opciones_modelos.append("SVM (Baseline)")

    if not opciones_modelos:
        opciones_modelos = ["Ningún modelo disponible"]

    modelo_nombre = st.radio(
        "Seleccionar modelo",
        opciones_modelos,
        index=0,
        help="Elija entre la Red Neuronal Convolucional 1D o el modelo clásico SVM.",
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
        "Analizar Ventana",
        use_container_width=True,
        type="primary",
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
# ── Layout de Contenido ───────────────────────────────────────────────────────
col_centro, col_derecha = st.columns([3.2, 1.3], gap="large")

ETIQUETAS_NOMBRES = {
    "Wake": "Vigilia",
    "N1": "Fase N1 (Ligero)",
    "N2": "Fase N2 (Intermedio)",
    "N3": "Fase N3 (Profundo)",
    "REM": "Fase REM (Sueño paradójico)",
}

if archivo is None:
    with col_centro:
        st.info("👈 Por favor cargue un archivo `*-PSG.edf` en el panel lateral para iniciar el análisis.")
        st.markdown(
            """
            #### Instrucciones:
            1. En la barra lateral, cargue un archivo de polisomnografía (por ejemplo: `SC4001E0-PSG.edf`).
            2. Seleccione el modelo deseado (**CNN 1D** o **SVM**).
            3. Navegue a través de las ventanas temporales de 30 segundos y pulse **Analizar Ventana**.
            """
        )

else:
    nombre_archivo = archivo.name.lower()
    es_hipnograma = "hypnogram" in nombre_archivo

    if es_hipnograma:
        with col_centro:
            st.error(
                f"⚠️ **Archivo de anotaciones detectado**: Has subido `{archivo.name}`, el cual es un archivo de hipnograma.\n\n"
                "Los hipnogramas contienen únicamente las etiquetas clínicas del especialista y carecen de canales continuos de EEG/EOG. "
                "Para visualizar señales y realizar inferencias con los modelos, por favor cargue el archivo PSG correspondiente "
                "(ejemplo: `SC4001E0-PSG.edf` en lugar de `SC4001EC-Hypnogram.edf`)."
            )
    else:
        try:
            raw = cargar_edf_bytes(archivo.getvalue())
            fs = float(raw.info["sfreq"])
            duracion = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
            max_ventanas = int(duracion // 30)

            canales_requeridos = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
            canales_presentes = [ch for ch in canales_requeridos if ch in raw.ch_names]

            with col_centro:
                st.subheader("Visualización de Señales Fisiológicas")

                if max_ventanas == 0:
                    st.warning("El archivo cargado dura menos de 30 segundos; no se pueden extraer ventanas completas.")
                    ventana = 0
                elif max_ventanas == 1:
                    st.caption("ℹ️ Archivo con una única ventana disponible de 30 segundos (0 a 30s).")
                    ventana = 0
                else:
                    ventana = st.slider(
                        "Seleccionar Ventana Temporal (30 segundos)",
                        min_value=0,
                        max_value=max_ventanas - 1,
                        value=0,
                        format="Ventana %d",
                    )

                inicio = ventana * 30
                start = int(inicio * fs)
                stop = int((inicio + 30) * fs)

                canales_visibles = []
                if mostrar_fpz:
                    canales_visibles.append("EEG Fpz-Cz")
                if mostrar_pz:
                    canales_visibles.append("EEG Pz-Oz")
                if mostrar_eog:
                    canales_visibles.append("EOG horizontal")

                if not canales_presentes:
                    st.error(
                        f"El archivo EDF no contiene los canales esperados ({canales_requeridos}). "
                        f"Canales en el archivo: {raw.ch_names}"
                    )
                else:
                    for canal in canales_visibles:
                        if canal in raw.ch_names:
                            signal = raw.get_data(
                                picks=[canal], start=start, stop=stop
                            )[0]
                            fig = grafico_signal(signal, fs, f"{canal} - Ventana {ventana} ({inicio}s - {inicio+30}s)")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                        else:
                            st.warning(f"Canal no disponible en el archivo: {canal}")

            with col_derecha:
                st.subheader("Diagnóstico del Modelo")

                if analizar:
                    if max_ventanas == 0:
                        st.error("No hay ventanas válidas para analizar.")
                    elif not canales_presentes:
                        st.error("Faltan canales requeridos para inferencia.")
                    else:
                        try:
                            if "CNN 1D" in modelo_nombre:
                                modelo_cnn = get_cnn_model()
                                if modelo_cnn is None:
                                    st.error(f"No se pudo cargar CNN 1D desde {CNN_MODEL_PATH}")
                                else:
                                    x_tensor = preparar_tensor_cnn(raw, inicio)
                                    etapa, probs = predecir_cnn(modelo_cnn, x_tensor)
                                    badge_txt = "🧠 Deep Learning (CNN 1D)"
                            else:
                                modelo_svm = get_svm_model()
                                if modelo_svm is None:
                                    st.error(f"No se pudo cargar SVM desde {SVM_MODEL_PATH}")
                                else:
                                    x_features = preparar_features(raw, inicio)
                                    etapa, probs = predecir(modelo_svm, x_features)
                                    badge_txt = "⚙️ Machine Learning (SVM)"

                            confianza = probs.get(etapa, 0.0)
                            desc_etapa = ETIQUETAS_NOMBRES.get(etapa, "")

                            st.markdown(
                                f"""
                                <div class="pred-card">
                                    <div class="model-badge">{badge_txt}</div>
                                    <div class="stage">{etapa}</div>
                                    <div class="stage-desc">{desc_etapa}</div>
                                    <div class="conf">{confianza:.1%} confianza</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown("#### Distribución de Probabilidad")
                            orden = ["Wake", "N1", "N2", "N3", "REM"]
                            for clase in orden:
                                p = probs.get(clase, 0.0)
                                col_lbl, col_pct = st.columns([3, 1])
                                with col_lbl:
                                    st.write(f"**{clase}** ({ETIQUETAS_NOMBRES.get(clase, '')})")
                                with col_pct:
                                    st.write(f"{p:.1%}")
                                st.progress(min(max(float(p), 0.0), 1.0))

                        except Exception as e:
                            st.error(f"Error durante la predicción: {e}")

                else:
                    st.info("💡 Selecciona una ventana y pulsa **Analizar Ventana** para clasificar la etapa de sueño.")

        except Exception as exc:
            with col_centro:
                st.error(f"Error al leer el archivo EDF: {exc}")
