from pathlib import Path
import sys
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.svm_model import crear_svm
from src.evaluation.metrics import calcular_metricas


# configuracion

MLFLOW_DB = ROOT / "mlflow.db"
DATASET = ROOT / "outputs" / "data" / "dataset_sleep_ml.csv"
MODEL_OUT = ROOT / "models" / "svm_sleep.joblib"
PRED_OUT = ROOT / "outputs" / "predictions" / "predicciones_svm.csv"
CM_OUT = ROOT / "outputs" / "metrics" / "confusion_matrix_svm.csv"
EXPERIMENT = "sleep-stage-svm"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")


# cargar datos

df = pd.read_csv(DATASET)

columnas_excluir = ["subject", "study", "archivo", "epoch", "stage"]

X = df.drop(columns=columnas_excluir)
y = df["stage"]
groups = df["subject"]

print("Dimensiones:", df.shape)
print("Sujetos:", groups.nunique())
print("\nClases:")
print(y.value_counts())


# dividir por sujeto

split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

subjects_train = groups.iloc[train_idx]
subjects_test = groups.iloc[test_idx]

print("\nSujetos train:")
print(subjects_train.unique())

print("\nSujetos test:")
print(subjects_test.unique())

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# crear carpetas

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
CM_OUT.parent.mkdir(parents=True, exist_ok=True)


# experimento mlflow

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run():

    # entrenar modelo

    modelo = crear_svm(C=1.0, kernel="rbf", gamma="scale")
    modelo.fit(X_train, y_train)

    # predecir

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    # metricas

    metricas = calcular_metricas(y_test, y_pred)

    mlflow.log_param("model", "SVM")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("gamma", "scale")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("train_subjects", subjects_train.nunique())
    mlflow.log_param("test_subjects", subjects_test.nunique())

    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)

    # guardar modelo

    joblib.dump(modelo, MODEL_OUT)
    mlflow.sklearn.log_model(modelo, "model")

    # guardar predicciones

    resultados = pd.DataFrame({
        "subject": subjects_test.values,
        "real": y_test.values,
        "predicho": y_pred
    })

    for i, clase in enumerate(modelo.classes_):
        resultados[f"prob_{clase}"] = y_proba[:, i]

    resultados.to_csv(PRED_OUT, index=False)

    # matriz de confusion

    clases = ["Wake", "N1", "N2", "N3", "REM"]

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=clases),
        index=clases,
        columns=clases
    )

    cm.to_csv(CM_OUT)

    # registrar artefactos

    mlflow.log_artifact(str(PRED_OUT))
    mlflow.log_artifact(str(CM_OUT))

    # resultados

    print("\nMetricas:")
    print(metricas)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nMatriz de confusion:")
    print(cm)

    print("\nModelo guardado en:")
    print(MODEL_OUT)

    print("\nPredicciones guardadas en:")
    print(PRED_OUT)