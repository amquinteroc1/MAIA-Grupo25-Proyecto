# MAIA Grupo 25 - Clasificación de Etapas del Sueño

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

## 3. Descargar los datos con DVC

Los datos originales no se almacenan directamente en Git.

```powershell
dvc pull
```

Después del `pull`, debe existir la carpeta:

```text
data/
```

con los registros de Sleep-EDF.

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

| Modelo | Arquitectura | Entrada | Comando de Ejecución | Artefacto Serializado | Experimento MLflow |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM** | Support Vector Machine (RBF) | 30 features (Espectrales / Temporales) | `python -m experiments.experiment_svm` | `models/svm_sleep.joblib` | `sleep-stage-svm` |
| **LightGBM** | Gradient Boosting Decision Trees | 30 features (Espectrales / Temporales) | `python -m experiments.experiment_lightgbm` | `models/lightgbm_sleep.joblib` | `sleep-stage-lightgbm` |
| **CNN 1D** | Red Neuronal Convolucional Profunda | Señal cruda 3 canales × 3000 muestras (100 Hz) | `python src/models/train_cnn.py` | `experiments/cnn1d/best_cnn1d.pt` | `sleep-stage-mlp-gpu` |

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

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Estado de salud y modelos activos en memoria |
| `GET` | `/api/v1/models` | Catálogo de modelos registrados y disponibilidad |
| `POST` | `/api/v1/edf/info` | Sube un EDF y obtiene duración, canales y cantidad de ventanas sin costo de inferencia |
| `POST` | `/api/v1/analyze/window` | **Análisis de ventana con selección múltiple**: Diagnóstico por modelo, probabilidades y consenso |
| `POST` | `/api/v1/analyze/batch` | Análisis por lotes para construcción de hipnograma secuencial |

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

## 10. Flujo completo

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
    └── API REST FastAPI (Swagger OpenAPI para integración)
```

---

## 11. Evaluación

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

## 12. Reproducibilidad

El flujo completo puede ejecutarse con:

```powershell
# 1. Descargar datos versionados
dvc pull

# 2. Construir dataset
python -m src.data.crear_dataset

# 3. Entrenar modelos
python -m experiments.experiment_svm
python -m experiments.experiment_lightgbm

# 4. Iniciar Dashboard interactivo
streamlit run dashboard/app.py

# 5. Iniciar API REST con Swagger OpenAPI
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
