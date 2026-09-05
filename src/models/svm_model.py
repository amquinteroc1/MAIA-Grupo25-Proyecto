from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# crear svm
def crear_svm(C=1.0, kernel="rbf", gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",
            probability=True
        ))
    ])