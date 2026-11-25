"""
schemas.py
Esquemas Pydantic v2 para validación, serialización y documentación OpenAPI (Swagger).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Catálogo de Modelos ───────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    """Información técnica y estado operativo de un clasificador."""
    id: str = Field(..., description="Identificador único del modelo (ej: 'svm', 'lightgbm', 'cnn1d')")
    name: str = Field(..., description="Nombre comercial/técnico legible")
    type: str = Field(..., description="Familia algorítmica (Machine Learning / Deep Learning)")
    path: str = Field(..., description="Ruta local al artefacto del modelo serializado")
    available: bool = Field(..., description="Indica si el modelo está disponible para inferencia")
    classes: List[str] = Field(..., description="Etapas del sueño soportadas")
    input_type: str = Field(..., description="Tipo de entrada requerida (features o señal cruda)")
    notes: Optional[str] = Field(None, description="Observaciones o dependencias necesarias")


class ModelsListResponse(BaseModel):
    """Respuesta del listado de modelos disponibles."""
    count: int = Field(..., description="Total de modelos registrados en el catálogo")
    available_count: int = Field(..., description="Total de modelos listos para inferencia")
    models: List[ModelInfo] = Field(..., description="Lista detallada de modelos")


# ── Metadatos del Archivo EDF ─────────────────────────────────────────────────
class EDFInfoResponse(BaseModel):
    """Metadatos de la señal polisomnográfica extraída del archivo EDF."""
    filename: str = Field(..., description="Nombre del archivo cargado")
    sampling_rate_hz: float = Field(..., description="Frecuencia de muestreo en Hertz (ej: 100 Hz)")
    duration_seconds: float = Field(..., description="Duración total del registro en segundos")
    duration_hours: float = Field(..., description="Duración total del registro en horas")
    total_windows_30s: int = Field(..., description="Número total de ventanas clínicas de 30 segundos")
    channels_present: List[str] = Field(..., description="Canales requeridos presentes en el archivo")
    channels_missing: List[str] = Field(..., description="Canales requeridos ausentes en el archivo")
    is_valid: bool = Field(..., description="True si contiene todos los canales para realizar inferencia")
    message: str = Field(..., description="Mensaje explicativo del estado del archivo")


class EDFUploadResponse(EDFInfoResponse):
    """Respuesta al registrar un archivo EDF en caché para navegación rápida sin re-subida."""
    edf_id: str = Field(..., description="Identificador único del EDF cacheado en el servidor")


class WindowSignalsResponse(BaseModel):
    """Señales fisiológicas en µV de una ventana de 30s listas para graficar en el frontend."""
    edf_id: str
    window_index: int
    start_second: float
    end_second: float
    sampling_rate_hz: float
    time_axis: List[float]
    channels: Dict[str, List[float]]


class CachedWindowAnalyzeRequest(BaseModel):
    """Petición JSON ligera para inferir una ventana sobre un EDF ya subido."""
    window_index: int = Field(0, ge=0)
    models: List[str] = Field(default_factory=lambda: ["e0", "e1", "e2"])


class TimelinePoint(BaseModel):
    """Punto muestreado del mapa rápido de estadios del sueño a lo largo de la noche."""
    window_index: int
    stage: str


class EDFTimelineResponse(BaseModel):
    """Mapa de color completo de estadios detectados a lo largo de todo el archivo EDF."""
    edf_id: str
    total_windows_30s: int
    model_used: str
    points: List[TimelinePoint]
    stage_counts: Dict[str, int]
    execution_time_ms: float


# ── Predicción y Análisis de una Ventana ───────────────────────────────────────
class ModelPrediction(BaseModel):
    """Diagnóstico individual emitido por un modelo para una ventana de 30s."""
    model_id: str = Field(..., description="ID del modelo ejecutor ('svm', 'lightgbm', 'cnn1d')")
    model_name: str = Field(..., description="Nombre formal del modelo")
    stage: str = Field(..., description="Etapa predicha ('Wake', 'N1', 'N2', 'N3', 'REM')")
    stage_description: str = Field(..., description="Descripción clínica en español (ej: 'Vigilia')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de certidumbre/probabilidad máxima (0 a 1)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Distribución de probabilidad para cada una de las 5 fases canónicas"
    )


class ConsensusAnalysis(BaseModel):
    """Análisis comparativo y consenso entre los múltiples modelos seleccionados."""
    consensus_stage: str = Field(..., description="Etapa de sueño mayoritaria acordada")
    consensus_description: str = Field(..., description="Descripción clínica de la etapa de consenso")
    agreement: bool = Field(..., description="True si el 100% de los modelos seleccionados coincidieron")
    agreement_percentage: float = Field(..., ge=0.0, le=100.0, description="Porcentaje de coincidencia entre modelos")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza promedio de los modelos evaluados")


class WindowAnalysisResponse(BaseModel):
    """Respuesta integral del análisis multimodelo para una ventana temporal de 30s."""
    filename: str = Field(..., description="Nombre del archivo EDF analizado")
    window_index: int = Field(..., description="Índice de la ventana de 30 segundos (0-indexed)")
    start_second: float = Field(..., description="Segundo de inicio de la ventana en el registro")
    end_second: float = Field(..., description="Segundo de finalización de la ventana (start + 30s)")
    models_analyzed: List[str] = Field(..., description="Lista de IDs de modelos que generaron inferencias")
    predictions: Dict[str, ModelPrediction] = Field(
        ...,
        description="Diccionario de diagnósticos indexado por el ID de cada modelo"
    )
    consensus: ConsensusAnalysis = Field(..., description="Métricas de concordancia y consenso multimodelo")
    execution_time_ms: float = Field(..., description="Tiempo total de procesamiento en milisegundos")


# ── Análisis por Lotes / Hipnograma Completo ──────────────────────────────────
class HypnogramEpoch(BaseModel):
    """Fase predicha para una época individual en un hipnograma."""
    window_index: int = Field(..., description="Índice secuencial de la época")
    start_second: float = Field(..., description="Segundo de inicio de la época")
    predictions: Dict[str, str] = Field(..., description="Etapa predicha por cada modelo (ej: {'svm': 'N2'})")
    consensus_stage: str = Field(..., description="Etapa asignada por consenso")


class BatchAnalysisResponse(BaseModel):
    """Resultado del análisis de múltiples ventanas o hipnograma completo."""
    filename: str = Field(..., description="Nombre del archivo EDF analizado")
    total_windows_analyzed: int = Field(..., description="Cantidad de ventanas procesadas")
    models_analyzed: List[str] = Field(..., description="Modelos utilizados en el lote")
    epochs: List[HypnogramEpoch] = Field(..., description="Secuencia temporal de épocas predichas")
    stage_distribution_by_model: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Frecuencia acumulada de cada fase según cada modelo"
    )
    consensus_distribution: Dict[str, int] = Field(
        ...,
        description="Frecuencia acumulada de cada fase según el consenso"
    )
    execution_time_ms: float = Field(..., description="Tiempo total de ejecución del lote en ms")


# ── Health Check ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Estado general y metadatos de la API."""
    status: str = Field(..., description="Estado de salud ('online')")
    service: str = Field(..., description="Nombre del microservicio")
    version: str = Field(..., description="Versión de la API")
    available_models: List[str] = Field(..., description="Modelos listos para recibir peticiones")
    docs_url: str = Field(..., description="Ruta a la documentación interactiva Swagger")


# ── Kaggle Batch API ──────────────────────────────────────────────────────────
class KaggleKernelEntry(BaseModel):
    """Configuración de un kernel remoto en Kaggle."""
    model_id: str
    env_var: str
    slug: str
    kernel_dir: str


class KaggleConfigResponse(BaseModel):
    """Estado de configuración de credenciales y slugs de Kaggle Batch API."""
    configured: bool
    username: Optional[str] = None
    has_key: bool
    kernels: Dict[str, KaggleKernelEntry]


class KaggleKernelStatusResponse(BaseModel):
    """Estado de ejecución remota de un notebook en Kaggle."""
    slug: str
    status: str
    failure_message: Optional[str] = None
    kaggle_url: str


class KaggleKernelRunResponse(BaseModel):
    """Respuesta al disparar una ejecución batch remota en Kaggle."""
    slug: str
    ref: str
    status: str
    kaggle_url: str
    message: str


class KaggleOutputFile(BaseModel):
    """Archivo de salida descargado desde un notebook de Kaggle."""
    file: str
    size_bytes: int


class KaggleKernelOutputsResponse(BaseModel):
    """Predicciones, resúmenes JSON y archivos obtenidos de Kaggle."""
    slug: str
    output_dir: str
    file_count: int
    files: List[KaggleOutputFile]
    json_summaries: Dict[str, Any]
    kaggle_url: str
