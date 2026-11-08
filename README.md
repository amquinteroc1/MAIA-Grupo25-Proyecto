# MAIA Grupo 25 — Clasificación de Etapas del Sueño

## Efecto del contexto temporal en la clasificación automática del sueño

Proyecto de clasificación automática de etapas del sueño mediante Deep Learning, utilizando señales polisomnográficas de [Sleep-EDF Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/).

Se comparan tres arquitecturas neuronales con un protocolo experimental compartido:

- **E0:** CNN 1D sobre épocas individuales.
- **E1:** CNN 1D compartida con concatenación de embeddings de cinco épocas consecutivas.
- **E2:** CNN 1D compartida con una GRU para incorporar la secuencia temporal.

Cada modelo se entrena con las semillas `42`, `123` y `2026`, para un total de **nueve corridas**.

## 1. Datos y señales

| Subconjunto          | Registros | Sujetos | Uso                                |
| -------------------- | --------: | ------: | ---------------------------------- |
| Sleep-Cassette (SC)  |       153 |      78 | Entrenamiento, validación y prueba |
| Sleep-Telemetry (ST) |        44 |      22 | Evaluación externa                 |
| **Total**            |   **197** | **100** |                                    |

**Canales:** `EEG Fpz-Cz`, `EEG Pz-Oz` y `EOG horizontal`. **Frecuencia:** 100 Hz. **Época:** 30 segundos (3.000 muestras por canal).

| Código | Clase | Anotación original            |
| -----: | ----- | ----------------------------- |
|      0 | Wake  | Sleep stage W                 |
|      1 | N1    | Sleep stage 1                 |
|      2 | N2    | Sleep stage 2                 |
|      3 | N3    | Sleep stage 3 y Sleep stage 4 |
|      4 | REM   | Sleep stage R                 |

## 2. Identificación de sujetos y particiones

Los registros se agrupan por persona. El identificador del sujeto se obtiene a partir del nombre del archivo PSG:

```python
record_id = filename.removesuffix("-PSG.edf")
subject_id = record_id[:5]
```

Por ejemplo, `SC4001E0-PSG.edf` y `SC4002E0-PSG.edf` corresponden al sujeto `SC400` y se asignan a la misma partición.

La partición SC utiliza `GroupShuffleSplit` con semilla `42`. Los registros ST forman el conjunto de evaluación externa. La normalización Z-score por canal utiliza la media y la desviación estándar calculadas sobre SC-train; los parámetros se conservan en `data/processed/meta/norm_params.npz`.

### Distribución de sujetos, registros y épocas

| Partición  | Sujetos | Registros |   Épocas E0 | Épocas E1/E2 |
| ---------- | ------: | --------: | ----------: | -----------: |
| SC-train   |      49 |        96 |     120.175 |      119.364 |
| SC-val     |      13 |        25 |      35.149 |       34.981 |
| SC-test    |      16 |        32 |      40.145 |       39.961 |
| ST externo |      22 |        44 |      42.467 |       42.003 |
| **Total**  | **100** |   **197** | **237.936** |  **236.309** |

**E0:** épocas individuales. **E1/E2:** épocas centrales que cuentan con cinco épocas cronológicamente consecutivas del mismo registro.

Para comparar los tres modelos se utilizan las mismas épocas centrales: **39.961 en SC-test** y **42.003 en ST**. E0 se entrena con todas sus épocas SC-train y se evalúa sobre las épocas centrales comunes en validación, SC-test y ST.

### Distribución de clases en E0

| Partición  |   Wake |     N1 |     N2 |    N3 |    REM |   Total |
| ---------- | -----: | -----: | -----: | ----: | -----: | ------: |
| SC-train   | 38.906 | 13.229 | 43.797 | 8.279 | 15.964 | 120.175 |
| SC-val     | 14.068 |  3.742 | 10.656 | 2.044 |  4.639 |  35.149 |
| SC-test    | 12.967 |  4.551 | 14.679 | 2.716 |  5.232 |  40.145 |
| ST externo |  4.199 |  3.653 | 19.851 | 6.415 |  8.349 |  42.467 |

El archivo `data/processed/meta/split_info.json` contiene `conteos_e0` y `conteos_seq`, con el detalle por clase de cada partición.

## 3. Generación del dataset y los tensores

**Script:** `src/data_tesis/crear_dataset_tesis.py`.

Los EDF de entrada se organizan en:

```text
data/raw/sleep-cassette/
data/raw/sleep-telemetry/
```

Desde la raíz del repositorio:

```powershell
python -m src.data_tesis.crear_dataset_tesis --fase todo
```

El procesamiento empareja PSG e Hypnogram, asigna las particiones por sujeto, extrae épocas de 30 segundos, aplica el mapeo de clases, calcula los parámetros de normalización y construye las matrices de datos:

| Directorio             | Contenido                                                              | Dimensión de entrada |
| ---------------------- | ---------------------------------------------------------------------- | -------------------- |
| `data/processed/e0/`   | `X_{train,val,test,st}.npy`, etiquetas y grupos                        | `(N, 3, 3000)`       |
| `data/processed/seq/`  | `X_seq_{train,val,test,st}.npy`, etiquetas, grupos e índices centrales | `(N, 5, 3, 3000)`    |
| `data/processed/meta/` | Inventario, particiones, grupos y normalización                        | Metadatos            |

**Metadatos principales:** `norm_params.npz`, `split_info.json`, `group_mappings.json` e `inventory_records.csv`.

### Validación de datos

```powershell
python -m src.data_tesis.test_data_tesis
```

`src/data_tesis/test_data_tesis.py` verifica las particiones por persona, los identificadores, etiquetas, dimensiones, normalización, continuidad de las secuencias y correspondencia entre épocas centrales de E0 y E1/E2.

### Consultar los conteos por clase

```powershell
python -c "import json; from pathlib import Path; d=json.loads(Path('data/processed/meta/split_info.json').read_text(encoding='utf-8')); [print(t,p,d[t][p]['por_clase']) for t in ('conteos_e0','conteos_seq') for p in ('train','val','test','st')]"
```

## 4. Arquitecturas

Los modelos usan cuatro bloques convolucionales con filtros `32`, `64`, `128` y `256`, BatchNorm, ReLU y MaxPooling. El embedding denso tiene 512 características y el dropout es 0,5.

| Modelo | Script                   | Entrada           | Agregación                             |
| ------ | ------------------------ | ----------------- | -------------------------------------- |
| E0     | `src/models/cnn1d.py`    | `(N, 3, 3000)`    | Clasificación individual               |
| E1     | `src/models/cnn1d_e1.py` | `(N, 5, 3, 3000)` | Concatenación de cinco embeddings      |
| E2     | `src/models/cnn1d_e2.py` | `(N, 5, 3, 3000)` | GRU de una capa y 256 unidades ocultas |

En E1 y E2, la etiqueta objetivo corresponde a la época central de la secuencia de cinco épocas.

La configuración compartida se define en `src/config.py`.

## 5. Entorno de ejecución

Versiones comprobadas en el entorno local de desarrollo:

| Componente                     | Versión        |
| ------------------------------ | -------------- |
| PyTorch                        | `2.12.0+cu126` |
| scikit-learn                   | `1.8.0`        |
| CUDA de la compilación PyTorch | `12.6`         |
| GPU CUDA                       | Disponible     |

Crear el entorno e instalar las dependencias en Windows/PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Instalar además una distribución de PyTorch compatible con CUDA y la GPU disponible.

Comprobar versiones y aceleración:

```powershell
python -c "import torch, sklearn; print('PyTorch:', torch.__version__); print('scikit-learn:', sklearn.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

## 6. Entrenamiento

El entrenamiento se organiza en `src/training/train.py`, `src/training/trainer.py` y `src/training/dataset.py`.

| Parámetro                  | Valor                                |
| -------------------------- | ------------------------------------ |
| Semillas                   | `42`, `123`, `2026`                  |
| Optimizador                | Adam                                 |
| Learning rate              | `0.001`                              |
| Weight decay               | `0.0001`                             |
| Batch size                 | `128`                                |
| Máximo de épocas           | `30`                                 |
| Función de pérdida         | CrossEntropyLoss ponderada por clase |
| Scheduler                  | ReduceLROnPlateau                    |
| Selección del mejor modelo | F1-macro de validación               |
| Early stopping             | Paciencia de 10 épocas               |

Verificar los tensores y la configuración:

```powershell
python -m src.training.train --experiment E0 --seed 42 --dry-run
```

Entrenar las nueve corridas con CUDA, desde un directorio de experimentos preparado para nuevas ejecuciones:

```powershell
python -m src.training.train --experiment all --seed all
```

Entrenar una corrida específica:

```powershell
python -m src.training.train --experiment E2 --seed 42
```

Los checkpoints, predicciones, probabilidades, historial y grupos se guardan en `experiments/E0/seed*/`, `experiments/E1/seed*/` y `experiments/E2/seed*/`.

Cada checkpoint sigue el patrón `best_model_E2_seed42.pt`, adaptado al experimento y a la semilla.

Las corridas se registran en MLflow, mediante el experimento `sleep-staging-contexto-temporal` y la base de datos `mlruns.db`.

## 7. Evaluación SC-test y ST

Los resultados se calculan con Accuracy, Balanced Accuracy, F1-macro, F1 por clase, Cohen's Kappa y matrices de confusión.

### Resultados promedio de las tres semillas

| Modelo | Accuracy SC-test | F1-macro SC-test | Accuracy ST | F1-macro ST |   F1 N1 ST |
| ------ | ---------------: | ---------------: | ----------: | ----------: | ---------: |
| E0     |          78,36 % |           0,7601 |     65,26 % |      0,6090 |     0,3603 |
| E1     |          81,51 % |           0,7814 |     61,19 % |      0,5893 |     0,3282 |
| **E2** |      **82,34 %** |       **0,7888** | **67,81 %** |  **0,6354** | **0,3884** |

### F1-macro por semilla

| Modelo | Semilla | SC-test |     ST |
| ------ | ------: | ------: | -----: |
| E0     |      42 |  0,7563 | 0,6145 |
| E0     |     123 |  0,7614 | 0,6240 |
| E0     |    2026 |  0,7625 | 0,5886 |
| E1     |      42 |  0,7807 | 0,6117 |
| E1     |     123 |  0,7717 | 0,5834 |
| E1     |    2026 |  0,7917 | 0,5728 |
| E2     |      42 |  0,7886 | 0,6549 |
| E2     |     123 |  0,7870 | 0,6461 |
| E2     |    2026 |  0,7907 | 0,6053 |

E2 obtuvo el F1-macro promedio más alto tanto en SC-test como en la evaluación externa ST.

### Evaluación externa ST

```powershell
python -m src.evaluation.evaluacion_st
```

El script valida la correspondencia de las **42.003 épocas** y los grupos de ST entre las nueve corridas, recalcula las métricas y genera `outputs/metrics/evaluacion_st_resumen.csv`.

### Matrices de confusión

```powershell
python -m src.evaluation.generar_matrices_confusion
```

Las matrices de SC-test y ST por arquitectura y semilla se almacenan en `src/evaluation/imagenes/matrices_confusion/`.

## 8. Aplicaciones y herramientas del repositorio

| Herramienta  | Archivo                    | Uso                                                      |
| ------------ | -------------------------- | -------------------------------------------------------- |
| Streamlit    | `dashboard/app.py`         | Carga y visualización de EDF, ventanas y predicciones    |
| FastAPI      | `src/api/main.py`          | Endpoints para metadatos EDF y análisis por ventana/lote |
| Inferencia   | `src/inference/predict.py` | Carga y ejecución de SVM, LightGBM y CNN1D               |
| MLflow       | `src/config.py`            | Registro de parámetros, métricas y corridas              |
| DVC          | `data.dvc`                 | Metadatos de versionado del directorio `data/`           |
| Hugging Face | `subir_HF.py`              | Selección e inventario de artefactos de experimentos     |

Ejecutar Streamlit:

```powershell
streamlit run dashboard/app.py
```

Ejecutar la API:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: `http://localhost:8000/docs`.

## 9. Estructura principal

```text
MAIA-Grupo25-Proyecto/
├── data/
│   ├── raw/
│   │   ├── sleep-cassette/
│   │   └── sleep-telemetry/
│   └── processed/
│       ├── e0/
│       ├── seq/
│       └── meta/
├── src/
│   ├── config.py
│   ├── data_tesis/
│   │   ├── crear_dataset_tesis.py
│   │   └── test_data_tesis.py
│   ├── models/
│   │   ├── cnn1d.py
│   │   ├── cnn1d_e1.py
│   │   └── cnn1d_e2.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── evaluacion_st.py
│   │   └── generar_matrices_confusion.py
│   ├── inference/
│   │   └── predict.py
│   └── api/
│       └── main.py
├── experiments/
│   ├── E0/
│   ├── E1/
│   └── E2/
├── dashboard/
│   └── app.py
├── outputs/
│   └── metrics/
├── data.dvc
├── subir_HF.py
├── requirements.txt
└── README.md
```

## 10. Ejecución por orden

Ejecutar desde la raíz del repositorio:

```powershell
# 1. Construir el dataset y los tensores
python -m src.data_tesis.crear_dataset_tesis --fase todo

# 2. Validar los tensores y particiones
python -m src.data_tesis.test_data_tesis

# 3. Verificar una corrida
python -m src.training.train --experiment E0 --seed 42 --dry-run

# 4. Entrenar E0, E1 y E2 para las tres semillas
python -m src.training.train --experiment all --seed all

# 5. Recalcular métricas de evaluación externa ST
python -m src.evaluation.evaluacion_st

# 6. Generar matrices de confusión
python -m src.evaluation.generar_matrices_confusion
```

## 11. Equipo

Proyecto desarrollado por el **Grupo 25 del programa MAIA**.

Las contribuciones del equipo se documentan mediante los commits del repositorio y los artefactos experimentales.
