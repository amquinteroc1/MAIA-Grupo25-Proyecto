from pathlib import Path
import sys
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.svm_model import crear_svm
from src.evaluation.metrics import calcular_metricas


# ── Configuracion ─────────────────────────────────────────────────────────────

MLFLOW_DB = ROOT / "mlflow.db"
MODEL_OUT = ROOT / "models" / "svm_sleep.joblib"
PRED_OUT = ROOT / "outputs" / "predictions" / "predicciones_svm.csv"
CM_OUT = ROOT / "outputs" / "metrics" / "confusion_matrix_svm.csv"
CNN_PRED = ROOT / "outputs" / "predictions" / "predicciones_mlp_gpu.csv"
EXPERIMENT = "sleep-stage-svm"

# Auto-deteccion del dataset con mayor cobertura disponible (mayor tamano en disco)
DATASET_CANDIDATES = [
    p for p in [
        ROOT / "outputs" / "data" / "dataset_sleep_ml.parquet",
        ROOT / "outputs" / "data" / "dataset_sleep_lightgbm.parquet",
        ROOT / "outputs" / "data" / "dataset_sleep_ml.csv",
        ROOT / "outputs" / "data" / "dataset_sleep_lightgbm.csv",
    ]
    if p.exists()
]

if not DATASET_CANDIDATES:
    raise FileNotFoundError(
        "No se encontro ningun dataset en outputs/data/. "
        "Ejecuta 'python -m src.data.crear_dataset' primero."
    )

# Seleccionar el archivo con mayor tamano (mayor cantidad de datos y sujetos)
DATASET = max(DATASET_CANDIDATES, key=lambda p: p.stat().st_size)

print(f"Cargando dataset: {DATASET.name}")
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")


# ── Cargar datos ──────────────────────────────────────────────────────────────

df = pd.read_parquet(DATASET) if DATASET.suffix == ".parquet" else pd.read_csv(DATASET)

columnas_excluir = ["subject", "study", "archivo", "epoch", "stage"]
X = df.drop(columns=columnas_excluir)
y = df["stage"]

# Identificador de grabacion unico a 6 caracteres (ej. 'SC4041', 'SC4042')
registro_id = df["archivo"].astype(str).str[:6]

print("Dimensiones totales:", df.shape)
print("Grabaciones unicas:", registro_id.nunique())
print("Sujetos unicos (5 chars):", df["subject"].nunique())
print("\nDistribucion global de clases:")
print(y.value_counts())


# ── Particion Train / Test ────────────────────────────────────────────────────
# Opcion 1: Alinear con CNN (predicciones_mlp_gpu.csv) si existe y solapa
# Opcion 2: GroupShuffleSplit (20% test) por grabacion

modo_split = "GroupShuffleSplit (20%)"
train_idx, test_idx = None, None

if CNN_PRED.exists():
    df_cnn = pd.read_csv(CNN_PRED)
    test_cnn_subjs = set(df_cnn["subject"].unique())
    is_test_cnn = registro_id.isin(test_cnn_subjs)

    n_test_coincidentes = is_test_cnn.sum()
    n_train_disponibles = (~is_test_cnn).sum()

    if n_test_coincidentes > 0 and n_train_disponibles > 0:
        modo_split = "Alineado con CNN (predicciones_mlp_gpu.csv)"
        train_idx = np.where(~is_test_cnn)[0]
        test_idx = np.where(is_test_cnn)[0]
        print(f"\n[Split] {modo_split}")
        print(f"Grabaciones test coincidentes con CNN: {registro_id.iloc[test_idx].nunique()} / {len(test_cnn_subjs)}")

if train_idx is None:
    print(f"\n[Split] {modo_split}")
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups=registro_id))

X_train_full = X.iloc[train_idx]
y_train_full = y.iloc[train_idx]
subjects_train_full = registro_id.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]
subjects_test = registro_id.iloc[test_idx]

print("\n--- Conjunto de Test Completo ---")
print(f"Epocas de test: {len(X_test)}")
print(f"Grabaciones en test ({subjects_test.nunique()}): {sorted(subjects_test.unique())}")


# ── Submuestreo balanceado para Train (Optimizacion O(N^2) SVM) ────────────────
# SVC(kernel='rbf') tiene complejidad temporal O(N^2) - O(N^3).
# Para entrenar en ~2-4 minutos sin colapsar memoria, tomamos un subconjunto
# balanceado estratificado por clase de X_train_full.
MAX_PER_CLASS = 4000  # ~20,000 epocas totales de entrenamiento

train_subsample_idx = []
rng = np.random.RandomState(42)

for clase in y_train_full.unique():
    idx_clase = np.where(y_train_full == clase)[0]
    n_elegir = min(len(idx_clase), MAX_PER_CLASS)
    seleccion = rng.choice(idx_clase, size=n_elegir, replace=False)
    train_subsample_idx.extend(train_idx[seleccion])

train_subsample_idx = np.array(train_subsample_idx)
rng.shuffle(train_subsample_idx)

X_train = X.iloc[train_subsample_idx]
y_train = y.iloc[train_subsample_idx]
subjects_train = registro_id.iloc[train_subsample_idx]

print("\n--- Conjunto de Entrenamiento (Submuestreo Balanceado) ---")
print(f"Epocas de train seleccionadas: {len(X_train)} (de {len(X_train_full)} disponibles)")
print(f"Grabaciones representadas en train: {subjects_train.nunique()}")
print("Distribucion en train:")
print(y_train.value_counts())


# ── Crear carpetas ────────────────────────────────────────────────────────────

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
CM_OUT.parent.mkdir(parents=True, exist_ok=True)


# ── Experimento MLflow ────────────────────────────────────────────────────────

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run():

    print("\nEntrenando modelo SVM (RBF)...")
    modelo = crear_svm(C=1.0, kernel="rbf", gamma="scale")
    modelo.fit(X_train, y_train)

    print("Generando predicciones sobre el 100% del test set...")
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    # metricas
    metricas = calcular_metricas(y_test, y_pred)

    mlflow.log_param("model", "SVM")
    mlflow.log_param("dataset", DATASET.name)
    mlflow.log_param("split_mode", modo_split)
    mlflow.log_param("C", 1.0)
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("gamma", "scale")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("train_samples_full", len(X_train_full))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("train_subjects", subjects_train.nunique())
    mlflow.log_param("test_subjects", subjects_test.nunique())
    mlflow.log_param("max_per_class", MAX_PER_CLASS)

    for nombre, valor in metricas.items():
        mlflow.log_metric(nombre, valor)

    # guardar modelo
    joblib.dump(modelo, MODEL_OUT)
    try:
        mlflow.sklearn.log_model(sk_model=modelo, name="model")
    except TypeError:
        mlflow.sklearn.log_model(modelo, "model")

    # guardar predicciones
    clases_orden = list(modelo.classes_)
    dict_resultados = {
        "subject": subjects_test.values,
        "real": y_test.values,
        "predicho": y_pred
    }

    for i, clase in enumerate(clases_orden):
        dict_resultados[f"prob_{clase}"] = y_proba[:, i]

    resultados = pd.DataFrame(dict_resultados)
    resultados.to_csv(PRED_OUT, index=False)

    # matriz de confusion
    clases_reporte = ["Wake", "N1", "N2", "N3", "REM"]
    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=clases_reporte),
        index=clases_reporte,
        columns=clases_reporte
    )
    cm.to_csv(CM_OUT)

    # registrar artefactos
    mlflow.log_artifact(str(PRED_OUT))
    mlflow.log_artifact(str(CM_OUT))

    # resultados en consola
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES SVM")
    print("=" * 60)
    print("\nMetricas globales:")
    for k, v in metricas.items():
        print(f"  {k}: {v:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=clases_reporte, zero_division=0))

    print("Matriz de confusion:")
    print(cm)

    print(f"\nModelo guardado en: {MODEL_OUT}")
    print(f"Predicciones guardadas en: {PRED_OUT}")
    print(f"Matriz de confusion guardada en: {CM_OUT}")