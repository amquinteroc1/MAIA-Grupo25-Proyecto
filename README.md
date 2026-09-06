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
│   └── experiment_svm.py
│
├── models/
│
├── outputs/
│   ├── data/
│   ├── metrics/
│   └── predictions/
│
├── src/
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
│       └── svm_model.py
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

## 6. Entrenar modelo SVM

El modelo implementado utiliza:

- `StandardScaler`
- Kernel RBF
- `class_weight="balanced"`
- Separación train/test por sujeto

La separación por sujeto evita que ventanas pertenecientes a una misma persona aparezcan simultáneamente en entrenamiento y prueba.

Ejecutar:

```powershell
python -m experiments.experiment_svm
```

El modelo entrenado se guarda en:

```text
models/svm_sleep.joblib
```

Las predicciones se guardan en:

```text
outputs/predictions/predicciones_svm.csv
```

La matriz de confusión se guarda en:

```text
outputs/metrics/confusion_matrix_svm.csv
```

---

## 7. MLflow

Los experimentos se registran con MLflow.

Durante el desarrollo local se utiliza:

```text
mlflow.db
```

Para abrir la interfaz local de MLflow:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir en el navegador:

```text
http://127.0.0.1:5000
```

MLflow registra:

- Modelo
- Hiperparámetros
- Número de sujetos de entrenamiento
- Número de sujetos de prueba
- Accuracy
- Balanced Accuracy
- Macro F1
- Predicciones
- Matriz de confusión
- Modelo entrenado

Para la entrega final, MLflow será desplegado en una instancia AWS EC2.

---

## 8. Ejecutar dashboard

Ejecutar:

```powershell
streamlit run dashboard/app.py
```

Abrir en el navegador:

```text
http://localhost:8501
```

El dashboard permite:

- Cargar un archivo PSG EDF
- Visualizar EEG Fpz-Cz
- Visualizar EEG Pz-Oz
- Visualizar EOG horizontal
- Seleccionar una ventana de 30 segundos
- Ejecutar el modelo
- Mostrar la etapa predicha
- Mostrar las probabilidades por clase

---

## 9. Flujo completo

```text
Sleep-EDF
    ↓
DVC
    ↓
PSG + Hypnogram
    ↓
Ventanas de 30 segundos
    ↓
Extracción de features
    ↓
Dataset ML
    ↓
Split por sujeto
    ↓
SVM
    ↓
Evaluación
    ↓
MLflow
    ↓
Modelo entrenado
    ↓
Dashboard Streamlit
```

---

## 10. Evaluación

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

## 11. Reproducibilidad

El flujo completo puede ejecutarse con:

```powershell
dvc pull
python -m src.data.crear_dataset
python -m experiments.experiment_svm
streamlit run dashboard/app.py
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
