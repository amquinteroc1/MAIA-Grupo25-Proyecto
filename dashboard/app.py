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

# ── Estilos CSS Profesionales (Accesibilidad WCAG AA + Mobile-First) ─────────
st.markdown(
    """
    <style>
    :root {
        --bg-main: #ffffff;
        --bg-surface: #f8fafc;
        --bg-subtle: #f1f5f9;
        --border-color: #cbd5e1;
        --border-soft: #e2e8f0;
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #475569;
        --brand-primary: #123b63;
        --brand-hover: #1b4f84;
        --focus-ring: #0284c7;
    }

    /* ── Canvas y Header sin cajas negras ─────────────────────────────────── */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.96) !important;
        border-bottom: 1px solid var(--border-soft);
    }
    header[data-testid="stHeader"] * {
        color: var(--text-primary) !important;
    }

    /* ── Espaciado Mobile-First ───────────────────────────────────────────── */
    .block-container {
        padding: 1.1rem 0.9rem 2rem 0.9rem !important;
        max-width: 1400px;
    }
    @media (min-width: 768px) {
        .block-container {
            padding: 1.75rem 2rem 2.5rem 2rem !important;
        }
    }

    /* ── Sidebar limpio y de alto contraste ───────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-soft) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.25rem;
    }
    [data-testid="stSidebar"] h1 {
        color: var(--brand-primary) !important;
        font-weight: 800;
        font-size: 1.45rem !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1e3a5f !important;
        font-weight: 700;
        font-size: 1.05rem !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--text-primary);
    }

    /* ── File Uploader sin caja negra ─────────────────────────────────────── */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #94a3b8 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s ease, background-color 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--brand-primary) !important;
        background-color: #f0f9ff !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--text-secondary) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #e2e8f0 !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 38px !important;
    }

    /* ── Expander (Kaggle Batch) sin cabecera negra ───────────────────────── */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stExpander"] details summary {
        background-color: var(--bg-surface) !important;
        color: var(--brand-primary) !important;
        font-weight: 700 !important;
        padding: 0.9rem 1.1rem !important;
        min-height: 48px;
        border-bottom: 1px solid var(--border-soft);
    }
    [data-testid="stExpander"] details summary * {
        color: var(--brand-primary) !important;
    }

    /* ── Selectbox / Inputs sin fondo oscuro ──────────────────────────────── */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        min-height: 44px !important;
    }
    [data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* ── Alertas (info / success / warning / error) alto contraste WCAG AA ── */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid var(--border-soft) !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] code {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="notification"][kind="info"],
    [data-testid="stNotificationContentInfo"] {
        background-color: #eff6ff !important;
        color: #1e3a8a !important;
    }
    div[data-baseweb="notification"][kind="positive"],
    [data-testid="stNotificationContentSuccess"] {
        background-color: #f0fdf4 !important;
        color: #166534 !important;
    }

    /* ── Botones accesibles (contraste AAA blanco sobre azul clínico, >=44px) */
    .stButton > button {
        background-color: var(--brand-primary) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 44px !important;
        padding: 0.6rem 1.1rem !important;
        transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
    }
    .stButton > button *,
    .stButton > button p,
    .stButton > button span {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: var(--brand-hover) !important;
        box-shadow: 0 4px 12px rgba(18, 59, 99, 0.22);
    }
    .stButton > button:focus-visible,
    [data-baseweb="select"] > div:focus-within,
    summary:focus-visible {
        outline: 3px solid var(--focus-ring) !important;
        outline-offset: 2px !important;
    }

    /* ── Hero y Tarjetas Responsivas (Mobile-First Grid) ──────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: clamp(1.1rem, 2.5vw, 1.8rem);
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: clamp(1.35rem, 2.6vw, 2rem);
        font-weight: 800;
        color: var(--brand-primary);
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: clamp(0.92rem, 1.3vw, 1.02rem);
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.5;
    }
    .thesis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin: 1rem 0 1.25rem 0;
    }
    .thesis-card {
        background: #ffffff;
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }
    .thesis-card.highlight {
        border: 2px solid #0284c7;
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 100%);
    }
    .thesis-card h4 {
        margin: 0 0 6px 0;
        color: var(--brand-primary);
        font-size: 1.02rem;
        font-weight: 800;
    }
    .thesis-card p {
        margin: 0 0 8px 0;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }
    .metric-pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 4px;
        margin-top: 4px;
    }
    .stage-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0.75rem 0 1.25rem 0;
    }
    .stage-pill {
        padding: 6px 12px;
        border-radius: 999px;
        background: var(--bg-subtle);
        border: 1px solid var(--border-soft);
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    /* ── Tarjetas de Diagnóstico Clínico ──────────────────────────────────── */
    .consensus-card {
        padding: 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%);
        border: 1px solid #86efac;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .pred-card {
        padding: 16px;
        border-radius: 14px;
        background-color: var(--bg-surface);
        text-align: center;
        border: 1px solid var(--border-soft);
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .stage {
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        color: var(--brand-primary);
        line-height: 1.1;
    }
    .stage-desc {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-top: 4px;
    }
    .conf {
        font-size: clamp(1.1rem, 2vw, 1.35rem);
        color: #1e3a8a;
        font-weight: 700;
        margin-top: 6px;
    }
    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        background-color: #e2e8f0;
        color: #0f172a;
        margin-bottom: 8px;
    }
    .agreement-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
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
        padding: 5px 12px;
        background-color: var(--bg-subtle);
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        font-size: 0.84rem;
        color: var(--text-primary);
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Caché de Lectura EDF (Optimizado: preload=False para <5 MB RAM) ───────────
@st.cache_resource(max_entries=3)
def cargar_edf_archivo(nombre: str, tamano: int, _file_bytes: bytes):
    """Guarda el EDF en archivo temporal persistente y lo abre con preload=False (bajo consumo RAM)."""
    cache_dir = Path(tempfile.gettempdir()) / "sleep_ai_edf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{tamano}_{Path(nombre).name}"
    tmp_path = cache_dir / safe_name
    if not tmp_path.exists() or tmp_path.stat().st_size != tamano:
        tmp_path.write_bytes(_file_bytes)
    return mne.io.read_raw_edf(tmp_path, preload=False, verbose=False)


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

    # 3. Selección de Modelos (Tesis E0/E1/E2, Individual, Todos o Personalizado)
    st.markdown("### 🤖 Modelo Clasificador")

    c_e0 = "✅" if catalog.get("e0", {}).get("available") else "❌"
    c_e1 = "✅" if catalog.get("e1", {}).get("available") else "❌"
    c_e2 = "✅" if catalog.get("e2", {}).get("available") else "❌"
    c_lgb = "✅" if catalog.get("lightgbm", {}).get("available") else "❌"
    c_svm = "✅" if catalog.get("svm", {}).get("available") else "❌"
    c_cnn = "✅" if catalog.get("cnn1d", {}).get("available") else "⚠️"

    opciones_modo = [
        "🏆 Consenso Tesis (E0 + E1 + E2)",
        f"🚀 E2 — CNN 1D + GRU 5 épocas {c_e2}",
        f"🔗 E1 — CNN 1D + Concat 5 épocas {c_e1}",
        f"🧠 E0 — CNN 1D Época individual {c_e0}",
        "🌟 Todos los modelos disponibles (Consenso)",
        "🔀 Selección personalizada",
    ]

    modo_seleccion = st.radio(
        "Modo de evaluación",
        opciones_modo,
        index=0,
        help=(
            "🏆 'Consenso Tesis' ejecuta E0, E1 y E2 simultáneamente sobre el EDF "
            "con la normalización oficial de SC-train y contexto de 5 épocas."
        ),
    )

    # Resolución de modelos según la selección
    modelos_a_evaluar = []

    if "Consenso Tesis" in modo_seleccion:
        modelos_a_evaluar = [m for m in ["e0", "e1", "e2"] if catalog.get(m, {}).get("available", False)]
        st.info(f"🏆 Evaluando **{len(modelos_a_evaluar)} modelos de tesis**: {', '.join(m.upper() for m in modelos_a_evaluar)}")

    elif "E2 —" in modo_seleccion:
        modelos_a_evaluar = ["e2"]

    elif "E1 —" in modo_seleccion:
        modelos_a_evaluar = ["e1"]

    elif "E0 —" in modo_seleccion:
        modelos_a_evaluar = ["e0"]

    elif "Todos los modelos disponibles" in modo_seleccion:
        modelos_a_evaluar = [m for m in ["e0", "e1", "e2", "lightgbm", "svm", "cnn1d"] if catalog.get(m, {}).get("available", False)]
        st.info(f"✨ Evaluando **{len(modelos_a_evaluar)} modelos activos**: {', '.join(m.upper() for m in modelos_a_evaluar)}")

    elif "personalizada" in modo_seleccion:
        opciones_custom = {
            f"E2 (CNN+GRU) {c_e2}": "e2",
            f"E1 (CNN+Concat) {c_e1}": "e1",
            f"E0 (CNN Base) {c_e0}": "e0",
            f"LightGBM {c_lgb}": "lightgbm",
            f"SVM {c_svm}": "svm",
            f"CNN 1D previo {c_cnn}": "cnn1d",
        }
        default_keys = [k for k, v in opciones_custom.items() if v in ("e0", "e1", "e2") and catalog.get(v, {}).get("available")]
        custom_elegidos = st.multiselect(
            "Seleccionar clasificadores a incluir",
            list(opciones_custom.keys()),
            default=default_keys,
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
# Si no hay archivo cargado: vista de bienvenida a ancho completo (mobile-first)
if archivo is None:
    st.markdown(
        """
        <div class="hero-banner" role="region" aria-label="Bienvenida Monitor Clínico">
            <h1 class="hero-title">💤 Monitor Clínico de Polisomnografía AI</h1>
            <p class="hero-sub">
                Clasificación automática de estadios del sueño (estándar <b>R&amp;K / AASM</b>) con contexto temporal
                sobre señales crudas EEG/EOG a 100 Hz · <b>MAIA Grupo 25 — Universidad de los Andes</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("👈 Carga un archivo `*-PSG.edf` en el panel lateral para visualizar señales y clasificar ventanas de 30s, o consulta abajo las corridas precalculadas vía Kaggle Batch API.")

    st.markdown(
        """
        <div class="stage-pills" aria-label="Etapas del sueño soportadas">
            <span class="stage-pill">☀️ Wake · Vigilia</span>
            <span class="stage-pill">🌙 N1 · Sueño ligero</span>
            <span class="stage-pill">💤 N2 · Intermedio</span>
            <span class="stage-pill">🌊 N3 · Profundo (Delta)</span>
            <span class="stage-pill">✨ REM · Paradójico</span>
        </div>

        <div class="thesis-grid">
            <div class="thesis-card">
                <h4>🧠 E0 · CNN 1D Base</h4>
                <p>Clasifica cada época individual de 30 s (3×3.000) sin contexto temporal. Línea base del estudio.</p>
                <span class="metric-pill">SC-test F1: 0,7601</span>
                <span class="metric-pill">ST F1: 0,6090</span>
            </div>
            <div class="thesis-card">
                <h4>🔗 E1 · Concatenación (5 épocas)</h4>
                <p>CNN 1D compartida sobre 5 épocas consecutivas (150 s) con concatenación posicional fija.</p>
                <span class="metric-pill">SC-test F1: 0,7814</span>
                <span class="metric-pill">ST F1: 0,5893</span>
            </div>
            <div class="thesis-card highlight">
                <h4>🚀 E2 · CNN 1D + GRU (Mejor)</h4>
                <p>Integración recurrente con compuertas sobre 5 épocas. Retención cross-domain R = +0,92.</p>
                <span class="metric-pill">SC-test F1: 0,7888</span>
                <span class="metric-pill">ST F1: 0,6354</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    col_centro, col_derecha = st.columns([3.2, 1.4], gap="large")
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
            raw = cargar_edf_archivo(archivo.name, archivo.size, archivo.getvalue())
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


# ── Sección de Ejecución y Consulta Batch Remota (Kaggle API) ─────────────────
st.markdown("---")
with st.expander("⚡ Batch Remoto y Predicciones Precalculadas (Kaggle API)", expanded=True):
    from src.utils.kaggle_batch import (
        fetch_kernel_outputs,
        get_kaggle_config,
        get_kernel_status,
        trigger_kaggle_kernel,
    )

    k_cfg = get_kaggle_config()
    if k_cfg["configured"]:
        st.success(f"Kaggle Batch API conectada · Usuario: `{k_cfg['username']}`")
    else:
        st.info(
            "Diligencia `KAGGLE_USERNAME` y `KAGGLE_KEY` en tu archivo `.env` para habilitar "
            "el disparo remoto y descarga de predicciones desde Kaggle."
        )

    col_k1, col_k2, col_k3, col_k4 = st.columns([2.2, 1, 1, 1])
    with col_k1:
        k_model = st.selectbox(
            "Notebook / Modelo de Tesis en Kaggle",
            options=["all", "e2", "e1", "e0"],
            format_func=lambda x: (
                "TODOS (E0 + E1 + E2)"
                if x == "all"
                else f"{x.upper()} — {k_cfg['kernels'][x]['slug']}"
            ),
            key="kaggle_model_select",
        )
    with col_k2:
        btn_k_status = st.button("📡 Estado", use_container_width=True)
    with col_k3:
        btn_k_outputs = st.button("📥 Traer Outputs", use_container_width=True)
    with col_k4:
        btn_k_run = st.button("🚀 Ejecutar Batch", use_container_width=True)

    targets = ["e0", "e1", "e2"] if k_model == "all" else [k_model]

    if btn_k_status:
        try:
            with st.spinner("Consultando estado remoto en Kaggle..."):
                st_rows = [get_kernel_status(t) for t in targets]
            st.dataframe(pd.DataFrame(st_rows), use_container_width=True, hide_index=True)
        except Exception as k_err:
            st.error(f"No se pudo consultar el estado en Kaggle: {k_err}")

    if btn_k_outputs:
        try:
            with st.spinner("Descargando predicciones y métricas desde Kaggle..."):
                all_outs = {t: fetch_kernel_outputs(t) for t in targets}

            metric_rows = []
            ckpt_rows = []
            for t, out_res in all_outs.items():
                for fname, payload in out_res.get("json_summaries", {}).items():
                    if isinstance(payload, dict) and "seeds" in payload:
                        exp_name = payload.get("experiment", t.upper())
                        if "checkpoint_verified" in payload:
                            cinfo = payload["checkpoint_verified"]
                            ckpt_rows.append({
                                "Modelo": exp_name,
                                "Checkpoint": cinfo.get("file"),
                                "Tamaño (MB)": cinfo.get("size_mb"),
                                "Mejor Época": cinfo.get("epoch"),
                                "Val F1-Macro": round(float(cinfo.get("val_f1_macro", 0.0)), 4),
                                "Kaggle URL": out_res.get("kaggle_url"),
                            })
                        for s_key, s_data in payload["seeds"].items():
                            tm = s_data.get("test_metrics", {})
                            metric_rows.append({
                                "Modelo": exp_name,
                                "Semilla": int(s_key),
                                "Mejor Época": s_data.get("best_epoch"),
                                "Acc SC-test": round(float(tm.get("test_accuracy", 0.0)), 4),
                                "F1-Macro": round(float(tm.get("test_f1_macro", 0.0)), 4),
                                "F1-N1": round(float(tm.get("test_f1_n1", 0.0)), 4),
                                "Kappa": round(float(tm.get("test_kappa", 0.0)), 4),
                                "Inf. SC-test (s)": round(float(tm.get("inference_time_test_s", 0.0)), 2),
                                "VRAM Pico (MB)": round(float(tm.get("gpu_mem_peak_mb", 0.0)), 1),
                            })

            if ckpt_rows:
                st.markdown("#### ✅ Checkpoints Verificados en Kaggle")
                st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True, hide_index=True)
            if metric_rows:
                st.markdown("#### 📊 Resultados SC-test descargados vía Kaggle Batch API")
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

            with st.expander("Ver JSON crudo descargado de Kaggle", expanded=False):
                st.json(all_outs)
        except Exception as k_err:
            st.error(f"No se pudieron descargar los outputs de Kaggle: {k_err}")

    if btn_k_run:
        try:
            with st.spinner("Enviando ejecución batch remota a Kaggle..."):
                run_rows = [trigger_kaggle_kernel(t) for t in targets]
            st.success(f"Ejecución remota disparada para: {', '.join(t.upper() for t in targets)}")
            st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)
        except Exception as k_err:
            st.error(f"No se pudo disparar el kernel en Kaggle: {k_err}")

