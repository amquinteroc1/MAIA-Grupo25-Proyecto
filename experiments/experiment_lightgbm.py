from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import joblib
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold
)


# rutas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import calcular_metricas
from src.models.lightgbm_model import crear_lightgbm


# configuracion

SEED = 42

CLASES = [
    "Wake",
    "N1",
    "N2",
    "N3",
    "REM"
]

COLUMNAS_CONTROL = [
    "subject",
    "study",
    "archivo",
    "epoch",
    "stage"
]

DATASET = ROOT / "outputs" / "data" / "dataset_sleep_lightgbm.parquet"
MODEL_OUT = ROOT / "models" / "lightgbm_sleep.joblib"

PRED_OUT = (
    ROOT
    / "outputs"
    / "predictions"
    / "predicciones_lightgbm.csv"
)

CM_OUT = (
    ROOT
    / "outputs"
    / "metrics"
    / "confusion_matrix_lightgbm.csv"
)

REPORT_OUT = (
    ROOT
    / "outputs"
    / "metrics"
    / "classification_report_lightgbm.csv"
)

IMPORTANCE_OUT = (
    ROOT
    / "outputs"
    / "metrics"
    / "feature_importance_lightgbm.csv"
)

CV_OUT = (
    ROOT
    / "outputs"
    / "metrics"
    / "cv_results_lightgbm.csv"
)

SPLIT_OUT = (
    ROOT
    / "outputs"
    / "metrics"
    / "split_subjects_lightgbm.json"
)

MLFLOW_DB = ROOT / "mlflow.db"
EXPERIMENT = "sleep-stage-lightgbm"


# crear carpetas

def crear_carpetas():
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    CM_OUT.parent.mkdir(parents=True, exist_ok=True)


# cargar datos

def cargar_datos():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {DATASET}"
        )

    dataset = pd.read_parquet(DATASET)

    columnas_faltantes = [
        columna
        for columna in COLUMNAS_CONTROL
        if columna not in dataset.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            f"Columnas faltantes: {columnas_faltantes}"
        )

    X = dataset.drop(columns=COLUMNAS_CONTROL)
    y = dataset["stage"]
    groups = dataset["subject"]

    if X.isna().any().any():
        raise ValueError(
            "El dataset contiene valores faltantes"
        )

    if np.isinf(X).any().any():
        raise ValueError(
            "El dataset contiene valores infinitos"
        )

    return dataset, X, y, groups


# dividir por sujeto

def dividir_datos(X, y, groups):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    sujetos_train = set(
        groups_train.unique()
    )

    sujetos_test = set(
        groups_test.unique()
    )

    compartidos = (
        sujetos_train
        & sujetos_test
    )

    if compartidos:
        raise ValueError(
            f"Fuga de sujetos: {compartidos}"
        )

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "groups_train": groups_train,
        "groups_test": groups_test,
        "sujetos_train": sujetos_train,
        "sujetos_test": sujetos_test
    }


# espacio de hiperparametros

def espacio_parametros():
    return {
        "n_estimators": [
            200,
            400,
            600
        ],
        "learning_rate": [
            0.03,
            0.05,
            0.10
        ],
        "num_leaves": [
            15,
            31,
            63
        ],
        "max_depth": [
            -1,
            8,
            12
        ],
        "min_child_samples": [
            20,
            40,
            80
        ],
        "subsample": [
            0.8,
            1.0
        ],
        "colsample_bytree": [
            0.8,
            1.0
        ],
        "reg_alpha": [
            0.0,
            0.1,
            0.5
        ],
        "reg_lambda": [
            0.0,
            0.1,
            0.5
        ]
    }


# baseline

def evaluar_baseline(
    X_train,
    y_train,
    X_test,
    y_test
):
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        X_train,
        y_train
    )

    y_pred = baseline.predict(
        X_test
    )

    return calcular_metricas(
        y_test,
        y_pred
    )


# ajustar modelo

def ajustar_modelo(
    X_train,
    y_train,
    groups_train,
    n_iter,
    cv_splits
):
    modelo = crear_lightgbm()

    validacion = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=SEED
    )

    busqueda = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=espacio_parametros(),
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=validacion,
        refit=True,
        random_state=SEED,
        n_jobs=1,
        verbose=2,
        return_train_score=False
    )

    busqueda.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    return busqueda


# guardar split

def guardar_split(resultado):
    informacion = {
        "random_state": SEED,
        "test_size": 0.20,
        "n_train_subjects": len(
            resultado["sujetos_train"]
        ),
        "n_test_subjects": len(
            resultado["sujetos_test"]
        ),
        "train_subjects": sorted(
            resultado["sujetos_train"]
        ),
        "test_subjects": sorted(
            resultado["sujetos_test"]
        )
    }

    SPLIT_OUT.write_text(
        json.dumps(
            informacion,
            indent=2
        ),
        encoding="utf-8"
    )


# guardar predicciones

def guardar_predicciones(
    dataset,
    test_idx,
    y_test,
    y_pred,
    y_proba,
    clases_modelo
):
    resultados = (
        dataset
        .iloc[test_idx][
            [
                "subject",
                "study",
                "archivo",
                "epoch"
            ]
        ]
        .reset_index(drop=True)
    )

    resultados["real"] = (
        y_test.to_numpy()
    )

    resultados["predicho"] = y_pred

    for indice, clase in enumerate(
        clases_modelo
    ):
        resultados[
            f"prob_{clase}"
        ] = y_proba[:, indice]

    resultados.to_csv(
        PRED_OUT,
        index=False
    )


# guardar evaluacion

def guardar_evaluacion(
    y_test,
    y_pred
):
    matriz = pd.DataFrame(
        confusion_matrix(
            y_test,
            y_pred,
            labels=CLASES
        ),
        index=CLASES,
        columns=CLASES
    )

    matriz.index.name = "real"
    matriz.columns.name = "predicho"

    matriz.to_csv(CM_OUT)

    reporte = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=CLASES,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    reporte.to_csv(REPORT_OUT)

    return matriz, reporte


# guardar importancia

def guardar_importancia(
    modelo,
    columnas
):
    importance_gain = (
        modelo
        .booster_
        .feature_importance(
            importance_type="gain"
        )
    )

    importance_split = (
        modelo
        .booster_
        .feature_importance(
            importance_type="split"
        )
    )

    importancia = pd.DataFrame({
        "feature": columnas,
        "importance_gain": importance_gain,
        "importance_split": importance_split
    })

    total_gain = (
        importancia[
            "importance_gain"
        ].sum()
    )

    if total_gain > 0:
        importancia[
            "importance_gain_pct"
        ] = (
            importancia["importance_gain"]
            / total_gain
            * 100
        )

    else:
        importancia[
            "importance_gain_pct"
        ] = 0.0

    importancia = importancia.sort_values(
        "importance_gain",
        ascending=False
    )

    importancia.to_csv(
        IMPORTANCE_OUT,
        index=False
    )

    return importancia


# guardar validacion cruzada

def guardar_cv(busqueda):
    resultados = pd.DataFrame(
        busqueda.cv_results_
    )

    columnas = [
        "params",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_fit_time",
        "std_fit_time"
    ]

    resultados = (
        resultados[columnas]
        .sort_values(
            "rank_test_score"
        )
    )

    resultados.to_csv(
        CV_OUT,
        index=False
    )


# registrar mlflow

def registrar_mlflow(
    modelo_final,
    busqueda,
    metricas,
    metricas_baseline,
    resultado,
    n_iter,
    cv_splits
):
    parametros = {
        "model": "LightGBM",
        "search_method": "RandomizedSearchCV",
        "scoring": "balanced_accuracy",
        "n_iter": n_iter,
        "cv_splits": cv_splits,
        "class_weight": "balanced",
        "train_subjects": len(
            resultado["sujetos_train"]
        ),
        "test_subjects": len(
            resultado["sujetos_test"]
        )
    }

    parametros.update(
        busqueda.best_params_
    )

    mlflow.log_params(parametros)

    metricas_mlflow = {
        nombre: float(valor)
        for nombre, valor
        in metricas.items()
    }

    metricas_mlflow[
        "cv_balanced_accuracy"
    ] = float(
        busqueda.best_score_
    )

    for nombre, valor in (
        metricas_baseline.items()
    ):
        metricas_mlflow[
            f"baseline_{nombre}"
        ] = float(valor)

    mlflow.log_metrics(
        metricas_mlflow
    )

    # registrar primero los archivos

    for archivo in [
        PRED_OUT,
        CM_OUT,
        REPORT_OUT,
        IMPORTANCE_OUT,
        CV_OUT,
        SPLIT_OUT
    ]:
        mlflow.log_artifact(
            str(archivo)
        )

    # registrar modelo LightGBM

    mlflow.lightgbm.log_model(
        lgb_model=modelo_final,
        name="model",
        serialization_format="skops",
        skops_trusted_types=[
            "collections.OrderedDict",
            "lightgbm.basic.Booster",
            "lightgbm.sklearn.LGBMClassifier"
        ]
    )


# ejecutar experimento

def main(
    n_iter,
    cv_splits
):
    crear_carpetas()

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.as_posix()}"
    )

    mlflow.set_experiment(
        EXPERIMENT
    )

    dataset, X, y, groups = (
        cargar_datos()
    )

    resultado = dividir_datos(
        X,
        y,
        groups
    )

    X_train = resultado["X_train"]
    X_test = resultado["X_test"]

    y_train = resultado["y_train"]
    y_test = resultado["y_test"]

    groups_train = (
        resultado["groups_train"]
    )

    print(
        f"Dataset: {dataset.shape}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "Sujetos train: "
        f"{len(resultado['sujetos_train'])}"
    )

    print(
        "Sujetos test: "
        f"{len(resultado['sujetos_test'])}"
    )

    print(
        "Sujetos compartidos: 0"
    )

    print("\nClases train:")

    print(
        y_train.value_counts()
    )

    print("\nClases test:")

    print(
        y_test.value_counts()
    )

    guardar_split(resultado)

    with mlflow.start_run(
        run_name=f"random-search-{n_iter}"
    ):
        metricas_baseline = (
            evaluar_baseline(
                X_train,
                y_train,
                X_test,
                y_test
            )
        )

        busqueda = ajustar_modelo(
            X_train,
            y_train,
            groups_train,
            n_iter,
            cv_splits
        )

        modelo_final = (
            busqueda.best_estimator_
        )

        y_pred = modelo_final.predict(
            X_test
        )

        y_proba = (
            modelo_final.predict_proba(
                X_test
            )
        )

        metricas = calcular_metricas(
            y_test,
            y_pred
        )

        guardar_predicciones(
            dataset,
            resultado["test_idx"],
            y_test,
            y_pred,
            y_proba,
            modelo_final.classes_
        )

        matriz, reporte = (
            guardar_evaluacion(
                y_test,
                y_pred
            )
        )

        importancia = guardar_importancia(
            modelo_final,
            X.columns
        )

        guardar_cv(busqueda)

        joblib.dump(
            modelo_final,
            MODEL_OUT
        )

        registrar_mlflow(
            modelo_final,
            busqueda,
            metricas,
            metricas_baseline,
            resultado,
            n_iter,
            cv_splits
        )

        print(
            "\nMejores parametros:"
        )

        print(
            busqueda.best_params_
        )

        print(
            "\nBalanced Accuracy CV:"
        )

        print(
            busqueda.best_score_
        )

        print(
            "\nMetricas baseline:"
        )

        print(
            metricas_baseline
        )

        print(
            "\nMetricas LightGBM:"
        )

        print(
            metricas
        )

        print(
            "\nReporte:"
        )

        print(
            reporte
        )

        print(
            "\nMatriz de confusion:"
        )

        print(
            matriz
        )

        print(
            "\nTop 10 features:"
        )

        print(
            importancia.head(10)
        )

        print(
            "\nModelo guardado en:"
        )

        print(
            MODEL_OUT
        )


# argumentos

def obtener_argumentos():
    parser = ArgumentParser()

    parser.add_argument(
        "--n-iter",
        type=int,
        default=12
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3
    )

    return parser.parse_args()


# ejecutar

if __name__ == "__main__":
    args = obtener_argumentos()

    main(
        n_iter=args.n_iter,
        cv_splits=args.cv_splits
    )