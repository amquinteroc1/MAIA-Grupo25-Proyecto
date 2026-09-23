"""
main.py
Servidor FastAPI con documentación interactiva Swagger OpenAPI para análisis multimodelo de polisomnografía.
"""

from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchAnalysisResponse,
    EDFInfoResponse,
    HealthResponse,
    ModelInfo,
    ModelsListResponse,
    WindowAnalysisResponse,
)
from src.api.service import (
    analizar_lote_edf,
    analizar_ventana_edf,
    cargar_edf_desde_bytes,
    obtener_info_edf,
)
from src.inference.predict import get_model_catalog

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
        "description": "Clasificación de ventanas individuales de 30s e hipnogramas por lotes con selección múltiple.",
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


# ── Endpoints del Sistema ─────────────────────────────────────────────────────
@app.get(
    "/",
    tags=["Sistema"],
    summary="Información de bienvenida a la API",
)
def root():
    """Retorna información general de la API y enlaces a la documentación."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "swagger_docs": "/docs",
        "redoc_docs": "/redoc",
        "endpoints": {
            "models": "/api/v1/models",
            "edf_info": "/api/v1/edf/info",
            "analyze_window": "/api/v1/analyze/window",
            "analyze_batch": "/api/v1/analyze/batch",
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
        ["svm", "lightgbm"],
        description=(
            "Lista de modelos a ejecutar. Opciones: 'svm', 'lightgbm', 'cnn1d', 'all'. "
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
        ["svm", "lightgbm"],
        description="Modelos a evaluar en el lote. Opciones: 'svm', 'lightgbm', 'cnn1d', 'all'",
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

