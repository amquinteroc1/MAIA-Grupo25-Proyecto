"""
dashboard/app.py
Monitor del Sueño AI - Dashboard Clínico e Interactivo de Polisomnografía (EDF).
Soporta selección individual, múltiple y consenso de todos los modelos disponibles (LightGBM, SVM, CNN 1D).
"""

from collections import Counter
from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import streamlit as st

# ── Configuración de Rutas ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import (
    CANALES_ML,
    CLASES_CANONICAS,
    DESCRIPCION_ETAPAS,
    ejecutar_inferencia_modelo,
    get_model_catalog,
    preparar_features,
)

# ── Configuración de Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor del Sueño AI",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS Profesionales ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #172033;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] * {
        color: #172033;
    }
    [data-testid="stSidebar"] h1 {
        color: #123b63 !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #244b70 !important;
    }
    .stButton > button {
        background-color: #123b63 !important;
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #1b4f84 !important;
        box-shadow: 0 4px 8px rgba(18, 59, 99, 0.2);
    }
    .consensus-card {
        padding: 22px;
        border-radius: 14px;
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #bbf7d0;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .pred-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #f8fafc;
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stage {
        font-size: 46px;
        font-weight: 800;
        color: #123b63;
        line-height: 1.1;
    }
    .stage-desc {
        font-size: 15px;
        font-weight: 600;
        color: #475569;
        margin-top: 4px;
    }
    .conf {
        font-size: 22px;
        color: #1e3a8a;
        font-weight: 700;
        margin-top: 6px;
    }
    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background-color: #e2e8f0;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .agreement-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        margin-top: 8px;
    }
    .agreement-high {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .agreement-partial {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #fde047;
    }
    .agreement-single {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #bae6fd;
    }
    .meta-chip {
        display: inline-block;
        padding: 4px 12px;
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        font-size: 13px;
        color: #334155;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Caché de Lectura EDF ──────────────────────────────────────────────────────
@st.cache_resource
def cargar_edf_bytes(file_bytes: bytes):
    """Carga y parsea un archivo EDF en memoria con MNE."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        raw_edf = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        return raw_edf
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ── Graficador de Señal Fisiológica ───────────────────────────────────────────
def grafico_signal(sig_data: np.ndarray, sfreq: float, titulo: str):
    """Genera gráfico Matplotlib para un canal individual de polisomnografía."""
    tiempo = np.arange(len(sig_data)) / sfreq
    signal_uv = sig_data * 1e6  # Conversión a microvoltios

    fig_obj, ax = plt.subplots(figsize=(10, 2.0))
    ax.plot(tiempo, signal_uv, linewidth=0.85, color="#123b63")
    ax.set_title(titulo, fontsize=11, fontweight="600", color="#172033", pad=6)
    ax.set_xlabel("Segundos", fontsize=9, color="#475569")
    ax.set_ylabel("µV", fontsize=9, color="#475569")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_facecolor("#ffffff")
    fig_obj.patch.set_facecolor("#ffffff")
    fig_obj.tight_layout()
    return fig_obj


# ── Catálogo de Modelos del Sistema ───────────────────────────────────────────
catalog = get_model_catalog()
modelos_activos = [k for k, v in catalog.items() if v["available"]]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💤 Monitor del Sueño AI")
    st.caption("Clasificación de Polisomnografía · Sleep-EDF")

    st.markdown("---")

    # 1. Carga de archivo PSG
    st.markdown("### 📂 Archivo PSG")
    archivo = st.file_uploader(
        "Cargar archivo EDF",
        type=["edf"],
        help="Cargue un archivo *-PSG.edf con registros continuos de polisomnografía.",
    )

    # 2. Señales a visualizar
    st.markdown("### 📈 Señales a Visualizar")
    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        mostrar_fpz = st.checkbox("EEG Fpz-Cz", value=True)
        mostrar_pz = st.checkbox("EEG Pz-Oz", value=True)
    with col_ch2:
        mostrar_eog = st.checkbox("EOG horiz.", value=True)

    st.markdown("---")

    # 3. Selección de Modelos (Individual, Todos o Personalizado)
    st.markdown("### 🤖 Modelo Clasificador")

    c_lgb = "✅" if catalog.get("lightgbm", {}).get("available") else "❌"
    c_svm = "✅" if catalog.get("svm", {}).get("available") else "❌"
    c_cnn = "✅" if catalog.get("cnn1d", {}).get("available") else "⚠️ (Requiere PyTorch)"

    opciones_modo = [
        "🌟 Todos los modelos disponibles (Consenso)",
        f"🌲 LightGBM (Gradient Boosting) {c_lgb}",
        f"⚙️ SVM (Baseline) {c_svm}",
        f"🧠 CNN 1D (Deep Learning) {c_cnn}",
        "🔀 Selección personalizada",
    ]

    modo_seleccion = st.radio(
        "Modo de evaluación",
        opciones_modo,
        index=0,
        help=(
            "🌟 'Todos los modelos disponibles' ejecuta todos los clasificadores activos simultáneamente "
            "y calcula el consenso clínico inter-modelo."
        ),
    )

    # Resolución de modelos según la selección
    modelos_a_evaluar = []

    if "Todos los modelos disponibles" in modo_seleccion:
        modelos_a_evaluar = [m for m in ["lightgbm", "svm", "cnn1d"] if catalog.get(m, {}).get("available", False)]
        st.info(f"✨ Evaluando **{len(modelos_a_evaluar)} modelos activos**: {', '.join(m.upper() for m in modelos_a_evaluar)}")

    elif "LightGBM" in modo_seleccion:
        modelos_a_evaluar = ["lightgbm"]

    elif "SVM" in modo_seleccion:
        modelos_a_evaluar = ["svm"]

    elif "CNN 1D" in modo_seleccion:
        modelos_a_evaluar = ["cnn1d"]
        if not catalog["cnn1d"]["available"]:
            st.warning("⚠️ **CNN 1D no está disponible en este servidor.**")
            st.code("pip install torch", language="bash")
            st.caption("Ejecute el comando anterior en la terminal del servidor para habilitar CNN 1D.")

    elif "personalizada" in modo_seleccion:
        opciones_custom = {
            "LightGBM": "lightgbm",
            "SVM": "svm",
            "CNN 1D": "cnn1d",
        }
        custom_elegidos = st.multiselect(
            "Seleccionar clasificadores a incluir",
            list(opciones_custom.keys()),
            default=[k for k, v in [("LightGBM", "lightgbm"), ("SVM", "svm"), ("CNN 1D", "cnn1d")] if catalog.get(v, {}).get("available")],
            help="Seleccione 1, 2 o más modelos para comparar sus predicciones en la misma ventana.",
        )
        modelos_a_evaluar = [opciones_custom[c] for c in custom_elegidos]

    st.markdown("---")

    analizar = st.button(
        "🔬 Analizar Ventana",
        use_container_width=True,
        type="primary",
    )


# ── Layout de Contenido Principal ─────────────────────────────────────────────
col_centro, col_derecha = st.columns([3.2, 1.4], gap="large")

# Si no hay archivo cargado
if archivo is None:
    with col_centro:
        st.info("👈 Por favor cargue un archivo `*-PSG.edf` en el panel lateral para iniciar el análisis.")
        st.markdown(
            """
            ### Bienvenido al Monitor Clínico de Polisomnografía AI

            Este sistema permite diagnosticar y clasificar automáticamente las etapas de sueño según el estándar canónico **R&K / AASM**:
            - **Wake**: Vigilia
            - **N1**: Fase N1 (Sueño ligero / somnolencia)
            - **N2**: Fase N2 (Sueño intermedio con husos y complejos K)
            - **N3**: Fase N3 (Sueño de ondas lentas delta / profundo)
            - **REM**: Fase REM (Sueño paradójico con movimientos oculares rápidos)

            #### Modelos Soportados en el Proyecto:
            1. 🌲 **LightGBM (Gradient Boosting)**: Modelo ensemble de alta velocidad entrenado sobre 30 características espectrales y temporales.
            2. ⚙️ **SVM (Support Vector Machine)**: Clasificador de margen máximo con kernel RBF y balanceo de clases.
            3. 🧠 **CNN 1D (Convolutional Neural Network)**: Red neuronal profunda end-to-end sobre señales crudas a 100 Hz.
            4. 🌟 **Consenso Multimodelo**: Ejecución simultánea de todos los modelos con votación mayoritaria y concordancia inter-modelo.
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

            canales_presentes = [ch for ch in CANALES_ML if ch in raw.ch_names]
            canales_faltantes = [ch for ch in CANALES_ML if ch not in raw.ch_names]

            # ── Columna Centro: Visualización de Señales ───────────────────────
            with col_centro:
                st.subheader("Visualización de Señales Fisiológicas")

                # Chips de metadatos clínicos
                st.markdown(
                    f"""
                    <div>
                        <span class="meta-chip">📁 <b>Archivo:</b> {archivo.name}</span>
                        <span class="meta-chip">⚡ <b>Muestreo:</b> {fs:.0f} Hz</span>
                        <span class="meta-chip">⏱️ <b>Duración:</b> {duracion / 3600:.2f} h ({duracion:.0f} s)</span>
                        <span class="meta-chip">📊 <b>Total Ventanas 30s:</b> {max_ventanas}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if max_ventanas == 0:
                    st.warning("El archivo cargado dura menos de 30 segundos; no se pueden extraer ventanas completas.")
                    ventana = 0
                else:
                    ventana = st.slider(
                        "Seleccionar Ventana Temporal (30 segundos)",
                        min_value=0,
                        max_value=max(0, max_ventanas - 1),
                        value=0,
                        format="Ventana %d",
                        help="Deslice para desplazarse temporalmente por la noche de sueño en intervalos estándar de 30 segundos.",
                    )

                inicio = ventana * 30
                start = int(inicio * fs)
                stop = int((inicio + 30) * fs)

                # Formato de tiempo hh:mm:ss
                mins, secs = divmod(inicio, 60)
                hrs, mins = divmod(mins, 60)
                tiempo_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

                st.caption(f"📍 **Posición temporal:** {tiempo_str} ({inicio}s a {inicio + 30}s) · Ventana {ventana} de {max_ventanas - 1}")

                canales_visibles = []
                if mostrar_fpz:
                    canales_visibles.append("EEG Fpz-Cz")
                if mostrar_pz:
                    canales_visibles.append("EEG Pz-Oz")
                if mostrar_eog:
                    canales_visibles.append("EOG horizontal")

                if canales_faltantes:
                    st.error(f"Faltan canales requeridos en el archivo EDF: {canales_faltantes}")
                elif not canales_visibles:
                    st.warning("Seleccione al menos una señal en la barra lateral para visualizar.")
                else:
                    for canal in canales_visibles:
                        if canal in raw.ch_names:
                            signal = raw.get_data(picks=[canal], start=start, stop=stop)[0]
                            fig = grafico_signal(signal, fs, f"{canal} · Ventana {ventana} ({inicio}s - {inicio+30}s)")
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

            # ── Columna Derecha: Diagnóstico e Inferencia ──────────────────────
            with col_derecha:
                st.subheader("Diagnóstico del Modelo")

                if not modelos_a_evaluar:
                    st.warning("No hay ningún modelo seleccionado para evaluar.")
                elif not analizar:
                    st.info("💡 Navegue a la ventana deseada y pulse **Analizar Ventana** para ejecutar el diagnóstico.")
                elif canales_faltantes:
                    st.error("No se puede clasificar: faltan canales requeridos en el EDF.")
                else:
                    try:
                        with st.spinner("Ejecutando inferencia..."):
                            predicciones = {}
                            features_df = None

                            # Optimización: si se evalúan modelos tabulares, extraer features solo 1 vez
                            modelos_tabulares = [m for m in modelos_a_evaluar if m in ("svm", "lightgbm")]
                            if modelos_tabulares:
                                features_df = preparar_features(raw, inicio=inicio, duracion=30.0)

                            for m_key in modelos_a_evaluar:
                                if m_key == "cnn1d" and not catalog["cnn1d"]["available"]:
                                    st.warning("⚠️ CNN 1D omitido: Requiere PyTorch.")
                                    continue
                                res = ejecutar_inferencia_modelo(
                                    m_key,
                                    raw,
                                    inicio=inicio,
                                    duracion=30.0,
                                    features_df=features_df,
                                )
                                predicciones[m_key] = res

                        if not predicciones:
                            if "cnn1d" in modelos_a_evaluar and not catalog["cnn1d"]["available"]:
                                st.error("No se pudo ejecutar CNN 1D porque PyTorch no está instalado en este entorno.")
                                st.info("💡 Ejecute `pip install torch` en la terminal del servidor para habilitarlo.")
                            else:
                                st.error("No se pudo obtener predicciones con los modelos seleccionados.")
                        else:
                            # ── Caso A: Múltiples Modelos (Consenso) ────────────
                            if len(predicciones) > 1:
                                etapas = [p["stage"] for p in predicciones.values()]
                                conteo = Counter(etapas)
                                etapa_consenso, max_votos = conteo.most_common(1)[0]
                                total_modelos = len(predicciones)
                                acuerdo_pct = (max_votos / total_modelos) * 100
                                desc_consenso = DESCRIPCION_ETAPAS.get(etapa_consenso, "")

                                badge_class = "agreement-high" if acuerdo_pct >= 75 else "agreement-partial"

                                st.markdown(
                                    f"""
                                    <div class="consensus-card">
                                        <div class="model-badge">🌟 Consenso Clínico Multimodelo</div>
                                        <div class="stage">{etapa_consenso}</div>
                                        <div class="stage-desc">{desc_consenso}</div>
                                        <div class="agreement-badge {badge_class}">
                                            🎯 {acuerdo_pct:.0f}% de Acuerdo ({max_votos}/{total_modelos} modelos)
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                st.markdown("#### Comparación por Modelo")
                                for m_key, p_res in predicciones.items():
                                    m_info = catalog.get(m_key, {})
                                    m_name = m_info.get("name", m_key.upper())
                                    st.markdown(
                                        f"""
                                        <div class="pred-card">
                                            <div class="model-badge">{m_name}</div>
                                            <div style="font-size: 28px; font-weight: 800; color: #123b63;">
                                                {p_res['stage']}
                                            </div>
                                            <div style="font-size: 13px; color: #475569;">
                                                {p_res['stage_description']}
                                            </div>
                                            <div class="conf" style="font-size: 18px;">
                                                {p_res['confidence']:.1%} confianza
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                # Gráfico comparativo de probabilidades
                                st.markdown("#### Distribución Comparativa de Probabilidad")
                                df_prob_comp = pd.DataFrame(
                                    {
                                        catalog.get(m_k, {}).get("name", m_k.upper()): [
                                            p_data["probabilities"].get(clase, 0.0) for clase in CLASES_CANONICAS
                                        ]
                                        for m_k, p_data in predicciones.items()
                                    },
                                    index=CLASES_CANONICAS,
                                )
                                st.bar_chart(df_prob_comp)

                            # ── Caso B: Modelo Individual ───────────────────────
                            else:
                                m_key, p_res = next(iter(predicciones.items()))
                                m_info = catalog.get(m_key, {})
                                m_name = m_info.get("name", m_key.upper())
                                etapa = p_res["stage"]
                                desc_etapa = p_res["stage_description"]
                                confianza = p_res["confidence"]
                                probs = p_res["probabilities"]

                                st.markdown(
                                    f"""
                                    <div class="pred-card">
                                        <div class="model-badge">🤖 {m_name}</div>
                                        <div class="stage">{etapa}</div>
                                        <div class="stage-desc">{desc_etapa}</div>
                                        <div class="conf">{confianza:.1%} confianza</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                st.markdown("#### Distribución de Probabilidad")
                                for clase in CLASES_CANONICAS:
                                    p = probs.get(clase, 0.0)
                                    col_lbl, col_pct = st.columns([3, 1])
                                    with col_lbl:
                                        st.write(f"**{clase}** ({DESCRIPCION_ETAPAS.get(clase, '')})")
                                    with col_pct:
                                        st.write(f"{p:.1%}")
                                    st.progress(min(max(float(p), 0.0), 1.0))

                    except Exception as e:
                        st.error(f"Error durante la inferencia: {e}")

        except Exception as exc:
            with col_centro:
                st.error(f"Error al leer el archivo EDF: {exc}")
