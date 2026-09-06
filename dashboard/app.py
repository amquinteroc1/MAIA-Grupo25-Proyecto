from pathlib import Path
import sys
import tempfile

import joblib
import mne
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# configuracion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import preparar_features, predecir

MODEL_PATH = ROOT / "models" / "svm_sleep.joblib"

st.set_page_config(
    page_title="Monitor del Sueño AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# estilos

st.markdown("""
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

h1, h2, h3 {
    color: #172033;
}

</style>
""", unsafe_allow_html=True)


# cargar modelo

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


# cargar edf

@st.cache_resource
def cargar_edf_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file_bytes)
        ruta = tmp.name

    return mne.io.read_raw_edf(ruta, preload=True, verbose=False)


# graficar señal

def grafico_signal(signal, fs, titulo):
    tiempo = np.arange(len(signal)) / fs
    signal_uv = signal * 1e6

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(tiempo, signal_uv, linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Segundos")
    ax.set_ylabel("µV")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig


# sidebar

with st.sidebar:
    st.title("Monitor del Sueño AI")

    st.markdown("### Señales")

    mostrar_fpz = st.checkbox("EEG Fpz-Cz", value=True)
    mostrar_pz = st.checkbox("EEG Pz-Oz", value=True)
    mostrar_eog = st.checkbox("EOG horizontal", value=True)

    archivo = st.file_uploader(
        "Cargar EDF",
        type=["edf"]
    )

    st.markdown("### Modelo")

    modelo_nombre = st.radio(
        "Seleccionar modelo",
        ["SVM"],
        index=0,
        label_visibility="collapsed"
    )

    analizar = st.button(
        "Analizar",
        use_container_width=True,
        type="primary"
    )


# layout

col_centro, col_derecha = st.columns(
    [3.2, 1.1],
    gap="large"
)


# contenido principal

if archivo is None:

    with col_centro:
        st.info("Carga un archivo PSG EDF para comenzar.")

else:

    raw = cargar_edf_bytes(
        archivo.getvalue()
    )

    fs = raw.info["sfreq"]
    duracion = raw.times[-1]
    max_ventanas = int(duracion // 30)

    with col_centro:

        st.subheader("Señales")

        ventana = st.slider(
            "Ventana de 30 segundos",
            min_value=0,
            max_value=max(0, max_ventanas - 1),
            value=0
        )

        inicio = ventana * 30
        start = int(inicio * fs)
        stop = int((inicio + 30) * fs)

        canales = []

        if mostrar_fpz:
            canales.append("EEG Fpz-Cz")

        if mostrar_pz:
            canales.append("EEG Pz-Oz")

        if mostrar_eog:
            canales.append("EOG horizontal")

        for canal in canales:

            if canal in raw.ch_names:

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

            else:
                st.warning(
                    f"Canal no disponible: {canal}"
                )

    with col_derecha:

        st.subheader("Predicción")

        if analizar:

            if not MODEL_PATH.exists():

                st.error(
                    f"No se encontró el modelo: {MODEL_PATH}"
                )

            else:

                try:

                    modelo = cargar_modelo()

                    X = preparar_features(
                        raw,
                        inicio
                    )

                    etapa, probs = predecir(
                        modelo,
                        X
                    )

                    confianza = probs.get(
                        etapa,
                        0
                    )

                    st.markdown(
                        f"""
                        <div class="pred-card">
                            <div class="stage">{etapa}</div>
                            <div class="conf">{confianza:.0%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("### Probabilidades")

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
                            f"{clase}: {prob:.1%}"
                        )

                        st.progress(
                            min(
                                max(float(prob), 0.0),
                                1.0
                            )
                        )

                except Exception as e:

                    st.error(
                        f"Error durante la predicción: {e}"
                    )

        else:

            st.info(
                "Selecciona una ventana y pulsa Analizar."
            )