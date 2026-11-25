"""
main.py
Servidor FastAPI con documentación interactiva Swagger OpenAPI para análisis multimodelo de polisomnografía.
"""

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    BatchAnalysisResponse,
    CachedWindowAnalyzeRequest,
    EDFInfoResponse,
    EDFTimelineResponse,
    EDFUploadResponse,
    HealthResponse,
    KaggleConfigResponse,
    KaggleKernelOutputsResponse,
    KaggleKernelRunResponse,
    KaggleKernelStatusResponse,
    ModelInfo,
    ModelsListResponse,
    WindowAnalysisResponse,
    WindowSignalsResponse,
)
from src.api.service import (
    analizar_lote_edf,
    analizar_ventana_edf,
    cargar_edf_desde_bytes,
    extraer_senales_ventana,
    generar_timeline_edf,
    obtener_edf_cacheado,
    obtener_info_edf,
    registrar_edf_en_cache,
)
from src.inference.predict import get_model_catalog
from src.utils.kaggle_batch import (
    fetch_kernel_outputs,
    get_kaggle_config,
    get_kernel_status,
    trigger_kaggle_kernel,
)

# ── Metadatos OpenAPI ─────────────────────────────────────────────────────────
API_TITLE = "API de Monitoreo y Clasificación del Sueño AI (Polisomnografía)"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Diagnóstico Automatizado de Fases de Sueño en Ventanas Clínicas de 30s

Esta API REST permite ejecutar el **mismo flujo de análisis que el dashboard interactivo de Streamlit**:
1. **Carga de Archivo EDF**: Sube registros de polisomnografía continua (`*-PSG.edf`).
2. **Selección Múltiple de Modelos**: Permite elegir qué clasificadores evaluar concurrentemente:
   - `svm`: Support Vector Machine con kernel RBF y ponderación balanceada.
   - `lightgbm`: Gradient Boosting multi-árbol de alta velocidad sobre 30 características espectrales/temporales.
   - `cnn1d`: Red Neuronal Convolucional 1D sobre señales crudas a 100 Hz (disponible con PyTorch).
   - `all`: Evalúa todos los modelos disponibles en el servidor simultáneamente.
3. **Diagnóstico Comparativo y Consenso**: Emite la fase predicha por cada modelo seleccionado, nivel de confianza, distribución de probabilidades de las 5 fases canónicas (Wake, N1, N2, N3, REM) y concordancia inter-modelo.
4. **Documentación Swagger OpenAPI**: Explora y prueba todos los endpoints interactivamente desde `/docs`.
"""

TAGS_METADATA = [
    {
        "name": "Modelos",
        "description": "Consulta de modelos registrados, arquitectura y disponibilidad técnica.",
    },
    {
        "name": "Polisomnografía (EDF)",
        "description": "Inspección de archivos EDF, metadatos, canales presentes y duración.",
    },
    {
        "name": "Análisis e Inferencia",
        "description": "Clasificación de ventanas individuales de 30s e hipnogramas por lotes con selección múltiple (E0, E1, E2, SVM, LightGBM, CNN 1D).",
    },
    {
        "name": "Batch Remoto (Kaggle)",
        "description": "Consulta de outputs/predicciones ya entrenadas y disparo remoto de notebooks en Kaggle Batch API.",
    },
    {
        "name": "Sistema",
        "description": "Verificación de salud y estado operativo del servicio.",
    },
]

# ── Instancia FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuración totalmente permisiva de CORS para saltar restricciones en localhost,
# IPs mockeadas (ej. http://34.233.121.225:8808), proxies y cualquier frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)



# ── Manejadores de Excepciones ────────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "ValidationError"},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_type": "ServiceUnavailable"},
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_UI_PATH = STATIC_DIR / "index.html"
FAVICON_PATH = STATIC_DIR / "assets" / "favicon.png"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Favicon y Frontend Web Stateless (Dashboard Clínico en FastAPI) ──────────
@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    if not FAVICON_PATH.is_file():
        raise HTTPException(status_code=404, detail="Favicon no encontrado.")
    return FileResponse(FAVICON_PATH, media_type="image/png")


@app.get(
    "/ui",
    tags=["Sistema"],
    summary="Dashboard Clínico Web Stateless (HTML5 + FastAPI)",
    include_in_schema=False,
)
def serve_web_ui():
    """Sirve el frontend web responsivo y stateless de polisomnografía."""
    if not STATIC_UI_PATH.is_file():
        raise HTTPException(status_code=404, detail="Interfaz web /ui no encontrada.")
    return FileResponse(STATIC_UI_PATH, media_type="text/html; charset=utf-8")


# ── Endpoints del Sistema ─────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Sistema"],
    summary="Información de bienvenida a la API",
)
def root():
    """Retorna información general de la API y enlaces al Dashboard /ui y Swagger."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "web_dashboard": "/ui",
        "swagger_docs": "/docs",
        "redoc_docs": "/redoc",
        "endpoints": {
            "web_ui": "/ui",
            "models": "/api/v1/models",
            "edf_upload": "/api/v1/edf/upload",
            "edf_info": "/api/v1/edf/info",
            "analyze_window": "/api/v1/analyze/window",
            "analyze_batch": "/api/v1/analyze/batch",
            "kaggle_config": "/api/v1/kaggle/config",
            "kaggle_status": "/api/v1/kaggle/status/{model_id}",
            "kaggle_outputs": "/api/v1/kaggle/outputs/{model_id}",
            "kaggle_run": "/api/v1/kaggle/run/{model_id}",
        },
    }


@app.get(
    "/health",
    tags=["Sistema"],
    summary="Verificar estado de salud de la API",
    response_model=HealthResponse,
)
def health_check():
    """Comprueba el funcionamiento del servicio y lista los modelos listos para inferencia."""
    catalog = get_model_catalog()
    disponibles = [k for k, v in catalog.items() if v["available"]]

    return HealthResponse(
        status="online",
        service=API_TITLE,
        version=API_VERSION,
        available_models=disponibles,
        docs_url="/docs",
    )


# ── Catálogo de Modelos ───────────────────────────────────────────────────────
@app.get(
    "/api/v1/models",
    tags=["Modelos"],
    summary="Listar modelos clasificadores disponibles",
    response_model=ModelsListResponse,
)
def list_models():
    """
    Retorna el inventario completo de modelos registrados (SVM, LightGBM, CNN 1D),
    indicando su estado de disponibilidad, ruta de artefactos y tipo de entrada.
    """
    catalog = get_model_catalog()
    models_list = [ModelInfo(**info) for info in catalog.values()]
    available_count = sum(1 for m in models_list if m.available)

    return ModelsListResponse(
        count=len(models_list),
        available_count=available_count,
        models=models_list,
    )


# ── Gestión Rápida de EDF Cacheado (Frontend Stateless /ui) ──────────────────
@app.post(
    "/api/v1/edf/upload",
    tags=["Polisomnografía (EDF)"],
    summary="Subir EDF una sola vez y obtener edf_id para navegación rápida",
    response_model=EDFUploadResponse,
)
async def upload_edf_cached(
    file: UploadFile = File(..., description="Archivo EDF de polisomnografía (*-PSG.edf)")
):
    """Guarda el EDF con preload=False (<5 MB RAM) y devuelve un edf_id reutilizable."""
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no admitido. Se requiere '.edf' (recibido: '{file.filename}').",
        )
    contenido = await file.read()
    try:
        return registrar_edf_en_cache(contenido, file.filename)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al registrar EDF: {exc}",
        )


@app.get(
    "/api/v1/edf/{edf_id}/window/{window_index}/signals",
    tags=["Polisomnografía (EDF)"],
    summary="Obtener señales fisiológicas en µV de una ventana de 30s",
    response_model=WindowSignalsResponse,
)
def get_cached_window_signals(edf_id: str, window_index: int):
    """Lee únicamente los 30s solicitados del EDF cacheado para graficar a 60 fps."""
    try:
        return extraer_senales_ventana(edf_id=edf_id, window_index=window_index)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))


@app.post(
    "/api/v1/edf/{edf_id}/analyze",
    tags=["Análisis e Inferencia"],
    summary="Ejecutar inferencia sobre un EDF ya cargado (sin re-subir archivo)",
    response_model=WindowAnalysisResponse,
)
def analyze_cached_window(edf_id: str, payload: CachedWindowAnalyzeRequest):
    """Clasifica la ventana seleccionada usando el EDF cacheado en disco."""
    try:
        filename, raw = obtener_edf_cacheado(edf_id)
        return analizar_ventana_edf(
            raw=raw,
            filename=filename,
            window_index=payload.window_index,
            model_ids=payload.models,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {exc}",
        )


@app.get(
    "/api/v1/edf/{edf_id}/timeline",
    tags=["Polisomnografía (EDF)"],
    summary="Mapa rápido de estadios del sueño sobre toda la línea de tiempo del EDF",
    response_model=EDFTimelineResponse,
)
def get_edf_timeline(edf_id: str, max_points: int = 240):
    """Escanea de forma vectorizada todo el estudio (~0.9 s) para pintar el hipnograma visual sobre el slider."""
    try:
        return generar_timeline_edf(edf_id=edf_id, max_points=max_points)
    except KeyError as k_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(k_err))
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando mapa de línea de tiempo: {exc}",
        )


# ── Inspección de Archivos EDF ────────────────────────────────────────────────
@app.post(
    "/api/v1/edf/info",
    tags=["Polisomnografía (EDF)"],
    summary="Inspeccionar metadatos y canales de un archivo EDF",
    response_model=EDFInfoResponse,
)
async def inspect_edf(
    file: UploadFile = File(
        ...,
        description="Archivo EDF de polisomnografía continua (*-PSG.edf)",
    )
):
    """
    Lee un archivo EDF y extrae su duración, frecuencia de muestreo y presencia
    de los canales fisiológicos requeridos (`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`)
    sin incurrir en costo de inferencia.
    """
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no admitido. Se requiere un archivo con extensión '.edf' (recibido: '{file.filename}').",
        )

    contenido = await file.read()
    try:
        raw = cargar_edf_desde_bytes(contenido, file.filename)
        info = obtener_info_edf(raw, file.filename)
        return info
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al leer el archivo EDF: {exc}",
        )


# ── Inferencia por Ventana (Equivalente al Dashboard) ──────────────────────────
@app.post(
    "/api/v1/analyze/window",
    tags=["Análisis e Inferencia"],
    summary="Analizar una ventana de 30s con selección múltiple de modelos",
    response_model=WindowAnalysisResponse,
)
async def analyze_window(
    file: UploadFile = File(
        ...,
        description="Archivo de polisomnografía en formato EDF (*-PSG.edf)",
    ),
    window_index: int = Form(
        0,
        ge=0,
        description="Índice de la ventana de 30 segundos (0 a N-1). Ejemplo: 0 para [0s - 30s], 1 para [30s - 60s].",
    ),
    models: List[str] = Form(
        ["e0", "e1", "e2"],
        description=(
            "Lista de modelos a ejecutar. Opciones: 'e0', 'e1', 'e2', 'svm', 'lightgbm', 'cnn1d', 'all'. "
            "En Swagger UI puedes añadir múltiples elementos o enviar varios valores."
        ),
    ),
):
    """
    **Endpoint principal de análisis como en el Dashboard:**
    1. Carga el archivo EDF en memoria.
    2. Extrae la ventana seleccionada de 30 segundos.
    3. Ejecuta la inferencia sobre los modelos seleccionados (SVM, LightGBM, CNN 1D).
    4. Retorna el diagnóstico individual, probabilidades y el análisis de consenso/acuerdo.
    """
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extensión inválida. Se requiere archivo '.edf' (recibido: '{file.filename}').",
        )

    contenido = await file.read()
    try:
        raw = cargar_edf_desde_bytes(contenido, file.filename)
        resultado = analizar_ventana_edf(
            raw=raw,
            filename=file.filename,
            window_index=window_index,
            model_ids=models,
        )
        return resultado
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el análisis de la ventana: {exc}",
        )


# ── Inferencia por Lotes / Hipnograma ──────────────────────────────────────────
@app.post(
    "/api/v1/analyze/batch",
    tags=["Análisis e Inferencia"],
    summary="Analizar múltiples ventanas consecutivas (Hipnograma)",
    response_model=BatchAnalysisResponse,
)
async def analyze_batch(
    file: UploadFile = File(
        ...,
        description="Archivo EDF de polisomnografía (*-PSG.edf)",
    ),
    start_window: int = Form(
        0,
        ge=0,
        description="Ventana inicial de 30 segundos (default: 0)",
    ),
    max_windows: int = Form(
        20,
        ge=1,
        le=100,
        description="Cantidad máxima de ventanas consecutivas a analizar (1 a 100)",
    ),
    models: List[str] = Form(
        ["e0", "e1", "e2"],
        description="Modelos a evaluar en el lote. Opciones: 'e0', 'e1', 'e2', 'svm', 'lightgbm', 'cnn1d', 'all'",
    ),
):
    """
    Analiza una secuencia temporal de ventanas de 30 segundos para generar un
    hipnograma multimodelo y la distribución acumulada de fases de sueño.
    """
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Extensión inválida. Se requiere archivo '.edf' (recibido: '{file.filename}').",
        )

    contenido = await file.read()
    try:
        raw = cargar_edf_desde_bytes(contenido, file.filename)
        resultado = analizar_lote_edf(
            raw=raw,
            filename=file.filename,
            start_window=start_window,
            max_windows=max_windows,
            model_ids=models,
        )
        return resultado
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el análisis del lote: {exc}",
        )


# ── Batch Remoto (Kaggle API) ────────────────────────────────────────────────
@app.get(
    "/api/v1/kaggle/config",
    tags=["Batch Remoto (Kaggle)"],
    summary="Verificar configuración de credenciales y slugs de Kaggle",
    response_model=KaggleConfigResponse,
)
def kaggle_config():
    """Retorna si KAGGLE_USERNAME / KAGGLE_KEY están configurados y los slugs de E0, E1 y E2."""
    return get_kaggle_config()


@app.get(
    "/api/v1/kaggle/status/{model_id}",
    tags=["Batch Remoto (Kaggle)"],
    summary="Consultar estado remoto de un notebook en Kaggle",
    response_model=KaggleKernelStatusResponse,
)
def kaggle_status(model_id: str):
    """Consulta el estado actual (`running`, `complete`, `error`) de un kernel Kaggle ('e0', 'e1', 'e2')."""
    try:
        return get_kernel_status(model_id)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except RuntimeError as run_err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(run_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error Kaggle API: {exc}")


@app.get(
    "/api/v1/kaggle/outputs/{model_id}",
    tags=["Batch Remoto (Kaggle)"],
    summary="Descargar y consultar predicciones/outputs generados en Kaggle",
    response_model=KaggleKernelOutputsResponse,
)
def kaggle_outputs(model_id: str):
    """Descarga los outputs del notebook ya ejecutado en Kaggle y devuelve los resúmenes JSON."""
    try:
        return fetch_kernel_outputs(model_id)
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except RuntimeError as run_err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(run_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error Kaggle API: {exc}")


@app.post(
    "/api/v1/kaggle/run/{model_id}",
    tags=["Batch Remoto (Kaggle)"],
    summary="Disparar ejecución batch remota de un notebook en Kaggle",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=KaggleKernelRunResponse,
)
def kaggle_run(model_id: str):
    """Dispara una ejecución batch remota en GPU de Kaggle (`kernels_push`) para 'e0', 'e1' o 'e2'."""
    try:
        return trigger_kaggle_kernel(model_id)
    except (ValueError, FileNotFoundError) as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))
    except RuntimeError as run_err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(run_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error Kaggle API: {exc}")


# ── Punto de entrada para ejecución directa ───────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8808,
        reload=True,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )

