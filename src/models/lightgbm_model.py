from lightgbm import LGBMClassifier


# configuracion base

PARAMETROS_BASE = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "class_weight": "balanced",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True
}


# crear modelo

def crear_lightgbm(**parametros):
    configuracion = (
        PARAMETROS_BASE.copy()
    )

    configuracion.update(
        parametros
    )

    return LGBMClassifier(
        **configuracion
    )