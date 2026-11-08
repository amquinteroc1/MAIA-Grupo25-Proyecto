# MAIA Grupo 25 - Clasificación de Etapas del Sueño

<!-- INICIO: TESIS E0-E1-E2 2026 -->

## Tesis 2026: efecto del contexto temporal en la clasificación de estadios del sueño

**Estado:** entrenamiento y evaluación finalizados para E0, E1 y E2; resultados externos en
Sleep-Telemetry (ST). Los experimentos anteriores SVM/LightGBM/CNN1D se mantienen como
antecedentes del microproyecto y **no son directamente comparables** con estas métricas:
se utilizaron particiónes y condiciones de evaluación diferentes.

### Datos y protocolo experimental actual

- Fuente: PhysioNet Sleep-EDF Expanded, versiónes SC (Sleep-Cassette) y ST (Sleep-Telemetry).
- Entrada: EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal; 100 Hz; épocas de 30 segundos.
- Etiquetas: Wake, N1, N2, N3 y REM.
- División por sujeto SC: entrenamiento 49, validación 13, prueba 16; ST externo 22 sujetos.
- Evaluación SC-test: **39.961 épocas** comúnes para las tres arquitecturas.
- Evaluación ST: **42.003 épocas** comúnes para las tres arquitecturas.
- Semillas por modelo: `42`, `123` y `2026` (nueve entrenamientos independientes).

### Arquitecturas

| Modelo | Entrada y agregación temporal                                    |
| ------ | ---------------------------------------------------------------- |
| E0     | CNN 1D sobre una epoca de 30 s, sin contexto                     |
| E1     | CNN compartida sobre 5 épocas y concatenacion de embeddings      |
| E2     | CNN compartida sobre 5 épocas y una GRU para agregación temporal |

La ventana objetivo en E1 y E2 es la epoca central de las cinco consecutivas.
El extractor CNN tiene cuatro bloques convolucionales, con 32/64/128/256 filtros.
Los tres modelos usan un protocolo de entrenamiento común; consultar `src/config.py`.

### Resultados: promedio de tres semillas

| Modelo | Accuracy SC-test | F1-macro SC-test | Accuracy ST | F1-macro ST |   F1 N1 ST |
| ------ | ---------------: | ---------------: | ----------: | ----------: | ---------: |
| E0     |          78,36 % |           0,7601 |     65,26 % |      0,6090 |     0,3603 |
| E1     |          81,51 % |           0,7814 |     61,19 % |      0,5893 |     0,3282 |
| **E2** |      **82,34 %** |       **0,7888** | **67,81 %** |  **0,6354** | **0,3884** |

En SC-test, E1 y E2 superan a E0. En ST, E2 mantiene el mejor F1-macro,
mientras E1 obtiene un resultado inferior a E0. Son medias descriptivas;
no constituyen una prueba de significacion estadistica por sujeto.

### Comandos de reproducibilidad de la tesis

Ejecutar desde la raiz del repositorio, con las dependencias ya instaladas:

```powershell
# Revisar la partición y los datos preparados
python -m src.data_tesis.test_data_tesis

# Regenerar datos de tesis cuando se disponga de los EDF originales
python -m src.data_tesis.crear_dataset_tesis --fase todo

# Comprobar configuración y particiónes sin entrenar
python -m src.training.train --experiment E0 --seed 42 --dry-run

# Entrenar las nueve corridas (solo si es necesario)
python -m src.training.train --experiment all --seed all

# Recalcular evaluación ST y matrices desde predicciones ya guardadas
python -m src.evaluation.evaluacion_st
python -m src.evaluation.generar_matrices_confusion

# Inventario para una futura publicación en Hugging Face (sin subida)
python subir_HF.py --root . --folder revision-luigi --dry-run
```

Los artefactos locales se guardan en `experiments/E0/`, `experiments/E1/`,
`experiments/E2/`, `outputs/metrics/` y `src/evaluation/imagenes/`.
No es necesario reentrenar para reproducir las métricas a partir de las predicciones guardadas.

### Versionado de datos y artefactos

- `data.dvc` registra el directorio `data/` (399 archivos y 8,72 GB en el metadato
  versiónado de Git). El remote DVC configurado es `cnn-remote` en S3. Su
  sincronización **debe verificarse** con `dvc status -c` antes de hacer limpieza.
- Los checkpoints, matrices y predicciones de los nueve experimentos permanecen
  locales hasta que se confirme su copia en Hugging Face u otro almacenamiento.
- Git versióna código y metadatos, pero **no sustituye** `dvc push` ni el respaldo
  de los pesos.
- Los repositorios Hugging Face originales son históricos; los repositorios `-v2`
  todavía requieren creacion/permisos antes de documentarlos como disponibles.

### Componentes anteriores

Las secciones siguientes documentan el **microproyecto previo** (SVM, LightGBM,
CNN1D, API, dashboard y despliegue). Se conservan como referencia, pero sus
comandos, rutas y resultados pueden corresponder a una versión distinta de la tesis.

<!-- FIN: TESIS E0-E1-E2 2026 -->

Proyecto de Machine Learning para la clasificación automática de etapas del sueño a partir de señales polisomnográficas del dataset **Sleep-EDF Expanded** de PhysioNet.

El objetivo es clasificar ventanas de 30 segundos en cinco etapas:

- Wake
- N1
- N2
- N3
- REM

---

## Estructura del proyecto

```text
MAIA-Grupo25-Proyecto/
│
├── data/
│
├── dashboard/
│   └── app.py
│
├── experiments/
│   ├── experiment_svm.py
│   └── experiment_lightgbm.py
│
├── models/
│   ├── svm_sleep.joblib
│   └── lightgbm_sleep.joblib
│
├── outputs/
│   ├── data/
│   ├── metrics/
│   └── predictions/
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── service.py
│   │
│   ├── data/
│   │   ├── explore_sleep_edf.py
│   │   ├── crear_dataset.py
│   │   └── features.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── models/
│       ├── svm_model.py
│       └── lightgbm_model.py
│
├── tests/
│   └── test_api.py
│
├── data.dvc
├── requirements.txt
└── README.md
```

---

## 1. Crear entorno virtual

```powershell
python -m venv .venv
```

Activar el entorno en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

---

## 2.1 Configuración de entorno y Hugging Face Hub

El proyecto utiliza **Hugging Face Hub** como repositorio oficial de almacenamiento para datos y modelos bajo la organización `maia-grupo25`:

- **Dataset Repo:** `maia-grupo25/sleep-edfx-data` (almacena datos crudos y preprocesados)
- **Model Repo:** `maia-grupo25/sleep-edfx-models` (almacena modelos serializados y métricas)

### A. Crear la variable `HUGGING_FACE_API_KEY`

1. **Vía archivo `.env` (Recomendado):**
   Copia la plantilla y coloca tu token personal de Hugging Face (con rol `Write`):

   ```powershell
   Copy-Item .env.example .env
   ```

   Edita `.env`:

   ```env
   HUGGING_FACE_API_KEY=hf_tu_token_aqui
   HF_DATA_REPO=maia-grupo25/sleep-edfx-data
   HF_MODEL_REPO=maia-grupo25/sleep-edfx-models
   ```

2. **Vía PowerShell (Sesión actual):**

   ```powershell
   $env:HUGGING_FACE_API_KEY = "hf_tu_token_aqui"
   ```

3. **Vía PowerShell (Permanente de usuario):**

   ```powershell
   [System.Environment]::SetEnvironmentVariable('HUGGING_FACE_API_KEY', 'hf_tu_token_aqui', 'User')
   ```

4. **Vía Bash / Linux / macOS:**
   ```bash
   export HUGGING_FACE_API_KEY="hf_tu_token_aqui"
   ```

### B. Verificar la conexión con Hugging Face

```powershell
python -m src.utils.hf_hub verify
```

---

## 3. Descargar los datos y modelos con Hugging Face / DVC

Los artefactos pesados están versionados en Hugging Face Hub y DVC:

```powershell
# Descargar modelos directamente desde Hugging Face Hub
python -m src.utils.hf_hub pull-models

# Descargar datos desde Hugging Face Hub
python -m src.utils.hf_hub pull-data

# O utilizando DVC tradicional:
dvc pull
```

Después de la sincronización, existirá la carpeta `data/` con los registros y los modelos en `models/` y `experiments/cnn1d/`.

---

## 4. Construir el dataset de Machine Learning

El pipeline realiza los siguientes pasos:

1. Lee los archivos PSG.
2. Empareja cada PSG con su archivo Hypnogram.
3. Divide las señales en ventanas de 30 segundos.
4. Utiliza los canales:
   - EEG Fpz-Cz
   - EEG Pz-Oz
   - EOG horizontal
5. Extrae características estadísticas y espectrales.
6. Genera la variable objetivo:
   - Wake
   - N1
   - N2
   - N3
   - REM

Ejecutar:

```powershell
python -m src.data.crear_dataset
```

Salida:

```text
outputs/data/dataset_sleep_ml.csv
```

---

## 5. Características utilizadas

Para cada señal se calculan:

- Media
- Desviación estándar
- Mínimo
- Máximo
- Rango
- RMS
- Potencia delta
- Potencia theta
- Potencia alpha
- Potencia beta

Cada fila del dataset representa una ventana de 30 segundos.

---

## 6. Modelos y Experimentos

El proyecto implementa y evalúa tres enfoques de Machine Learning y Deep Learning para la clasificación de etapas de sueño en ventanas de 30 segundos, todos versionados con DVC y registrados en experimentos dedicados de MLflow:

### Resumen Comparativo de Modelos:

| Modelo       | Arquitectura                        | Entrada                                        | Comando de Ejecución                        | Artefacto Serializado             | Experimento MLflow     |
| :----------- | :---------------------------------- | :--------------------------------------------- | :------------------------------------------ | :-------------------------------- | :--------------------- |
| **SVM**      | Support Vector Machine (RBF)        | 30 features (Espectrales / Temporales)         | `python -m experiments.experiment_svm`      | `models/svm_sleep.joblib`         | `sleep-stage-svm`      |
| **LightGBM** | Gradient Boosting Decision Trees    | 30 features (Espectrales / Temporales)         | `python -m experiments.experiment_lightgbm` | `models/lightgbm_sleep.joblib`    | `sleep-stage-lightgbm` |
| **CNN 1D**   | Red Neuronal Convolucional Profunda | Señal cruda 3 canales × 3000 muestras (100 Hz) | `python src/models/train_cnn.py`            | `experiments/cnn1d/best_cnn1d.pt` | `sleep-stage-mlp-gpu`  |

---

### 6.1. Support Vector Machine (SVM Baseline)

Clasificador de margen máximo con ponderación balanceada para mitigar el desbalance de clases (Wake vs N1/N3/REM):

- **Preprocesamiento**: `StandardScaler`.
- **Hiperparámetros**: Kernel `rbf`, `C=1.0`, `gamma="scale"`, `class_weight="balanced"`.
- **Estrategia de Validación**: `GroupShuffleSplit` por sujeto (alineado con los 36 sujetos de prueba evaluados por la CNN para comparación directa).

**Comando de ejecución:**

```powershell
python -m experiments.experiment_svm
```

**Salidas y artefactos:**

- **Modelo entrenado**: `models/svm_sleep.joblib`
- **Predicciones del conjunto de prueba**: `outputs/predictions/predicciones_svm.csv`
- **Matriz de confusión**: `outputs/metrics/confusion_matrix_svm.csv`

---

### 6.2. LightGBM (Gradient Boosting Ensemble)

Modelo basado en árboles de decisión con optimización de hiperparámetros y validación cruzada estratificada por sujeto:

- **Preprocesamiento**: 30 features espectrales (potencia por bandas $\delta, \theta, \alpha, \beta$) y estadísticas temporales (media, std, min, max, rango, RMS).
- **Optimización**: `RandomizedSearchCV` con `StratifiedGroupKFold` (agrupado por sujeto para evitar fuga de datos entre splits).
- **Hiperparámetros explorados**: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `class_weight="balanced"`.

**Comandos de ejecución:**

```powershell
# Ejecución estándar:
python -m experiments.experiment_lightgbm

# Con argumentos personalizados de iteraciones y splits de validación cruzada:
python -m experiments.experiment_lightgbm --n-iter 12 --cv-splits 3
```

**Salidas y artefactos:**

- **Modelo entrenado**: `models/lightgbm_sleep.joblib`
- **Predicciones del conjunto de prueba**: `outputs/predictions/predicciones_lightgbm.csv`
- **Matriz de confusión**: `outputs/metrics/confusion_matrix_lightgbm.csv`
- **Importancia de características**: `outputs/metrics/feature_importance_lightgbm.csv`

---

### 6.3. Red Neuronal Convolucional 1D (CNN 1D Deep Learning)

Red neuronal profunda end-to-end que opera directamente sobre la señal fisiológica cruda multicanal a 100 Hz (3000 muestras por ventana de 30s) sin requerir extracción manual de características:

- **Arquitectura**: Convoluciones temporales 1D con BatchNorm, activación ReLU, MaxPool, regularización Dropout (0.5) y capas lineales densas.
- **Entrada**: Tensor de dimensiones `(batch_size, 3, 3000)` correspondientes a los canales `EEG Fpz-Cz`, `EEG Pz-Oz` y `EOG horizontal`.
- **Optimizador y Pérdida**: AdamW con `ReduceLROnPlateau` y `CrossEntropyLoss` con boost ponderado en clases minoritarias ($N1$ y $N3$).

**Comandos de ejecución:**

- **En entorno local o servidor con PyTorch / GPU:**

  ```powershell
  # Ejecución por defecto:
  python src/models/train_cnn.py

  # Con hiperparámetros de entrenamiento:
  python src/models/train_cnn.py --epochs 50 --batch_size 128 --lr 1e-3
  ```

- **En Google Colab con aceleración GPU (NVIDIA T4 / V100):**
  Ejecutar el notebook interactivo con carga dinámica por memmap:
  ```text
  notebooks/train_cnn1d.ipynb
  ```

**Salidas y artefactos:**

- **Modelo serializado PyTorch**: `experiments/cnn1d/best_cnn1d.pt` (y `experiments/cnn1d/best_model.pth`)
- **Predicciones sobre los 36 sujetos de test**: `outputs/predictions/predicciones_mlp_gpu.csv`
- **Matriz de confusión**: `outputs/metrics/confusion_matrix_cnn1d.csv`

---

## 7. Seguimiento de Experimentos con MLflow

Todos los experimentos (SVM, LightGBM y CNN 1D) se registran de forma centralizada en la base de datos de tracking local:

```text
mlflow.db
```

### Iniciar la interfaz web de MLflow:

```powershell
# 1. Activar entorno virtual
.\.venv\Scripts\activate

# 2. Iniciar servidor local de tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### Experimentos registrados:

1. `sleep-stage-svm`: Experimentos del clasificador SVM (parámetros `C`, `gamma`, métricas de clasificación y matriz de confusión).
2. `sleep-stage-lightgbm`: Búsqueda de hiperparámetros y corridas de LightGBM (importancia de características y métricas de generalización).
3. `sleep-stage-mlp-gpu` / `sleep-stage-cnn-pipeline`: Entrenamiento y evaluación de la CNN 1D sobre los 36 sujetos del conjunto de prueba.

### Métricas registradas por corrida:

- Accuracy global
- Balanced Accuracy
- Macro F1-Score y F1 por etapa canónica (Wake, N1, N2, N3, REM)
- Matriz de confusión interactiva
- Artefactos serializados del modelo entrenado

---

## 8. Ejecutar dashboard

Para ejecutar el dashboard clínico interactivo:

```powershell
# 1. Asegurar activación del entorno virtual
.\.venv\Scripts\activate

# 2. Iniciar Streamlit
streamlit run dashboard/app.py
```

Abrir en el navegador:

```text
http://localhost:8501
```

El dashboard permite:

- Cargar un archivo PSG EDF (`*-PSG.edf`).
- Visualizar simultáneamente los canales fisiológicos continuos (**EEG Fpz-Cz**, **EEG Pz-Oz**, **EOG horizontal**).
- Navegar a lo largo del registro nocturno en ventanas temporales de 30 segundos.
- **Seleccionar todos los modelos disponibles (Consenso)**: Ejecuta simultáneamente **LightGBM**, **SVM** (y **CNN 1D**) para calcular el consenso clínico y el porcentaje de acuerdo inter-modelo.
- **Selección individual o personalizada**: Permite elegir un clasificador específico (**LightGBM**, **SVM**, **CNN 1D**) o comparar cualquier combinación deseada.
- Tarjetas de diagnóstico con la etapa canónica predicha (Wake, N1, N2, N3, REM) y descripción clínica.
- Gráficos comparativos de distribución de probabilidad para cada modelo en la ventana analizada.

---

## 9. Ejecutar API REST (FastAPI + Swagger OpenAPI)

La API REST permite realizar el análisis de polisomnografía de forma programática con **selección múltiple de modelos** (`svm`, `lightgbm`, `cnn1d` o `all`) y soporte para ventanas individuales o hipnogramas completos por lotes.

### Iniciar el servidor:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Documentación interactiva:

Abrir en el navegador para probar los endpoints interactivamente con Swagger UI:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### Endpoints principales:

| Método | Endpoint                 | Descripción                                                                                       |
| :----- | :----------------------- | :------------------------------------------------------------------------------------------------ |
| `GET`  | `/health`                | Estado de salud y modelos activos en memoria                                                      |
| `GET`  | `/api/v1/models`         | Catálogo de modelos registrados y disponibilidad                                                  |
| `POST` | `/api/v1/edf/info`       | Sube un EDF y obtiene duración, canales y cantidad de ventanas sin costo de inferencia            |
| `POST` | `/api/v1/analyze/window` | **Análisis de ventana con selección múltiple**: Diagnóstico por modelo, probabilidades y consenso |
| `POST` | `/api/v1/analyze/batch`  | Análisis por lotes para construcción de hipnograma secuencial                                     |

### Ejemplo de consumo desde Python (`requests`):

```python
import requests

url = "http://127.0.0.1:8000/api/v1/analyze/window"
files = {"file": open("data/sleep-cassette/SC4001E0-PSG.edf", "rb")}
data = {
    "window_index": 0,
    "models": ["svm", "lightgbm"]  # Selección múltiple
}

response = requests.post(url, files=files, data=data)
res = response.json()

print(f"Consenso: {res['consensus']['consensus_stage']} ({res['consensus']['agreement_percentage']}% de acuerdo)")
for m_id, pred in res["predictions"].items():
    print(f"  [{pred['model_name']}]: {pred['stage']} ({pred['confidence']:.1%} confianza)")
```

### Ejemplo con cURL:

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/analyze/window" `
     -F "file=@data/sleep-cassette/SC4001E0-PSG.edf" `
     -F "window_index=0" `
     -F "models=svm" `
     -F "models=lightgbm"
```

---

## 10. Despliegue con Docker y Docker Compose

El proyecto incluye contenedores Docker optimizados para los tres componentes clave de la solución, permitiendo un despliegue unificado, reproducible e independiente del sistema operativo anfitrión:

1. **`mlflow`**: Servidor centralizado de tracking y registro de experimentos (puerto `5000`).
2. **`api`**: Servidor REST FastAPI con documentación interactiva Swagger OpenAPI y soporte de inferencia con los 3 modelos (puerto `8000`).
3. **`dashboard`**: Aplicación interactiva de Streamlit para exploración de señales y diagnóstico clínico multimodelo (puerto `8501`).

### Estructura de archivos Docker:

- `Dockerfile.api`: Imagen ligera (`python:3.11-slim`) con Uvicorn, FastAPI, PyTorch CPU y dependencias de inferencia.
- `Dockerfile.dashboard`: Imagen ligera (`python:3.11-slim`) con Streamlit, visualización fisiológica y soporte multimodelo.
- `Dockerfile.mlflow`: Servidor de tracking con persistencia en SQLite (`mlflow.db`) y directorio de artefactos (`mlruns/`).
- `docker-compose.yml`: Orquestador con red interna (`sleep-network`), volúmenes persistentes de modelos y orden de arranque con `depends_on`.
- `.dockerignore`: Exclusión de datasets pesados, entornos virtuales y cachés para construcción ultrarrápida.

### Comandos de ejecución:

#### A. Levantar todos los servicios con Docker Compose (Recomendado):

```powershell
# 1. Construir y levantar los 3 contenedores en segundo plano:
docker compose up -d --build

# 2. Verificar el estado y chequeos de salud:
docker compose ps

# 3. Inspeccionar logs en tiempo real:
docker compose logs -f

# 4. Ver logs de un servicio específico (ej: api o dashboard):
docker compose logs -f api
```

#### B. URLs de acceso una vez levantados los servicios:

- **Dashboard Streamlit**: [http://localhost:8501](http://localhost:8501)
- **API Swagger OpenAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Servidor MLflow Tracking**: [http://localhost:5000](http://localhost:5000)

#### C. Detener los servicios:

```powershell
# Detener contenedores manteniendo datos y modelos persistentes:
docker compose down
```

#### D. Construir y ejecutar contenedores individualmente:

```powershell
# 1. API:
docker build -f Dockerfile.api -t sleep-api .
docker run -d -p 8000:8000 --name sleep-api sleep-api

# 2. Dashboard:
docker build -f Dockerfile.dashboard -t sleep-dashboard .
docker run -d -p 8501:8501 --name sleep-dashboard sleep-dashboard

# 3. MLflow:
docker build -f Dockerfile.mlflow -t sleep-mlflow .
docker run -d -p 5000:5000 --name sleep-mlflow sleep-mlflow
```

---

## 11. Flujo completo

```text
Sleep-EDF
    ↓
DVC
    ↓
PSG + Hypnogram
    ↓
Ventanas de 30 segundos
    ↓
Extracción de features / Señales crudas
    ↓
Modelos Clasificadores (SVM, LightGBM, CNN 1D)
    ↓
Evaluación y MLflow
    ↓
Interfaces de Usuario y Despliegue
    ├── Dashboard Streamlit (Visualización interactiva)
    ├── API REST FastAPI (Swagger OpenAPI para integración)
    └── Orquestación Docker Compose (Despliegue de microservicios)
```

---

## 12. Evaluación

Las principales métricas utilizadas son:

- Accuracy
- Balanced Accuracy
- Macro F1
- Precision
- Recall
- F1 por clase
- Matriz de confusión

Debido al desbalance entre las etapas del sueño, se priorizan **Balanced Accuracy** y **Macro F1** sobre la Accuracy global.

---

## 13. Reproducibilidad

El flujo completo puede ejecutarse con:

```powershell
# 1. Descargar datos y modelos versionados
python -m src.utils.hf_hub pull-models
python -m src.utils.hf_hub pull-data
# o alternativamente: dvc pull

# 2. Construir dataset (con opción de subir a Hugging Face)
python -m src.data.crear_dataset_cnn --push-hf
python -m src.data.crear_dataset_lightgbm --push-hf

# 3. Entrenar modelos y versionar en Hugging Face + DVC
python -m experiments.experiment_svm --push-hf
python -m experiments.experiment_lightgbm --push-hf
python -m src.models.train_cnn --push-hf

# 4. Sincronizar todos los modelos generados manualmente (opcional)
python -m src.utils.hf_hub push-models

# 5. Iniciar Dashboard interactivo
streamlit run dashboard/app.py

# 6. Iniciar API REST con Swagger OpenAPI
python -m uvicorn src.api.main:app --port 8000 --reload
```

---

## Datos

Dataset:

**Sleep-EDF Database Expanded**

Fuente:

**PhysioNet**

Se utilizan registros de:

- Sleep Cassette
- Sleep Telemetry

---

## Equipo

Proyecto desarrollado por el **Grupo 25** del curso MAIA.

Los aportes individuales se documentan mediante:

- Commits en el repositorio Git
- Reporte de trabajo en equipo
