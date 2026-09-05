MAIA Grupo 25 - Clasificación de Etapas del Sueño

Proyecto de Machine Learning para clasificación automática de etapas del sueño a partir de señales polisomnográficas del dataset Sleep-EDF Expanded de PhysioNet.

El objetivo es clasificar ventanas de 30 segundos en cinco etapas:

Wake

N1

N2

N3

REM

Estructura del proyecto

MAIA-Grupo25-Proyecto/
│
├── data/
├── dashboard/
│ └── app.py
├── experiments/
│ └── experiment_svm.py
├── models/
├── outputs/
│ ├── data/
│ ├── metrics/
│ └── predictions/
├── src/
│ ├── data/
│ │ ├── explore_sleep_edf.py
│ │ ├── crear_dataset.py
│ │ └── features.py
│ ├── evaluation/
│ │ └── metrics.py
│ ├── inference/
│ │ └── predict.py
│ └── models/
│ └── svm_model.py
├── data.dvc
├── requirements.txt
└── README.md

1. Crear entorno virtual

python -m venv .venv

Activar en Windows:

.\.venv\Scripts\Activate.ps1

2. Instalar dependencias

pip install -r requirements.txt

3. Descargar los datos con DVC

dvc pull

Después del pull debe existir la carpeta data/ con los registros Sleep-EDF.

4. Construir el dataset de Machine Learning

El pipeline:

Lee archivos PSG.

Empareja cada PSG con su Hypnogram.

Divide las señales en ventanas de 30 segundos.

Utiliza EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal.

Extrae características estadísticas y espectrales.

Genera la variable objetivo Wake, N1, N2, N3 o REM.

Ejecutar:

python -m src.data.crear_dataset

Salida:

outputs/data/dataset_sleep_ml.csv

5. Características utilizadas

Para cada señal se calculan:

media

desviación estándar

mínimo

máximo

rango

RMS

potencia delta

potencia theta

potencia alpha

potencia beta

Cada fila representa una ventana de 30 segundos.

6. Entrenar modelo SVM

El modelo utiliza:

StandardScaler

kernel RBF

class_weight="balanced"

separación train/test por sujeto

Ejecutar:

python -m experiments.experiment_svm

Salidas:

models/svm_sleep.joblib
outputs/predictions/predicciones_svm.csv
outputs/metrics/confusion_matrix_svm.csv

7. MLflow

Para abrir MLflow local:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Abrir:

http://127.0.0.1:5000

MLflow registra hiperparámetros, métricas, predicciones, matriz de confusión y modelo entrenado.

8. Ejecutar dashboard

streamlit run dashboard/app.py

Abrir:

http://localhost:8501

El dashboard permite cargar un PSG EDF, visualizar señales, seleccionar una ventana de 30 segundos, ejecutar el modelo y mostrar la etapa predicha y sus probabilidades.

9. Flujo completo

Sleep-EDF
↓
DVC
↓
PSG + Hypnogram
↓
ventanas de 30 segundos
↓
extracción de features
↓
dataset ML
↓
split por sujeto
↓
SVM
↓
evaluación
↓
MLflow
↓
modelo entrenado
↓
dashboard Streamlit

10. Evaluación

Métricas principales:

Accuracy

Balanced Accuracy

Macro F1

Precision

Recall

F1 por clase

matriz de confusión

Debido al desbalance entre etapas, se priorizan Balanced Accuracy y Macro F1.

11. Reproducibilidad

dvc pull
python -m src.data.crear_dataset
python -m experiments.experiment_svm
streamlit run dashboard/app.py

Datos

Sleep-EDF Database Expanded — PhysioNet.

Se utilizan registros de:

Sleep Cassette

Sleep Telemetry

Equipo

Proyecto desarrollado por el Grupo 25 del curso MAIA.

Los aportes individuales se documentan mediante commits en el repositorio Git y mediante el reporte de trabajo en equipo.
