"""
test_api.py
Suite de pruebas de integración para la API REST de Clasificación de Sueño.
Verifica endpoints de salud, catálogo, metadatos EDF y análisis multimodelo.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

SAMPLE_PSG = ROOT / "data" / "sleep-cassette" / "SC4001E0-PSG.edf"
SAMPLE_HYP = ROOT / "data" / "sleep-cassette" / "SC4001EC-Hypnogram.edf"

client = TestClient(app)


def test_root_endpoint():
    """Prueba el endpoint raíz /"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "swagger_docs" in data
    assert data["swagger_docs"] == "/docs"


def test_health_endpoint():
    """Prueba el health check /health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "available_models" in data
    assert "svm" in data["available_models"]
    assert "lightgbm" in data["available_models"]


def test_models_catalog_endpoint():
    """Prueba el listado de modelos /api/v1/models"""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 3
    model_ids = [m["id"] for m in data["models"]]
    assert "svm" in model_ids
    assert "lightgbm" in model_ids
    assert "cnn1d" in model_ids


def test_edf_info_endpoint():
    """Prueba la inspección de metadatos de un archivo EDF real."""
    if not SAMPLE_PSG.exists():
        print("Aviso: Archivo de prueba PSG no encontrado, omitiendo test")
        return

    with open(SAMPLE_PSG, "rb") as f:
        response = client.post(
            "/api/v1/edf/info",
            files={"file": (SAMPLE_PSG.name, f, "application/octet-stream")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == SAMPLE_PSG.name
    assert data["sampling_rate_hz"] == 100.0
    assert data["total_windows_30s"] > 0
    assert data["is_valid"] is True
    assert "EEG Fpz-Cz" in data["channels_present"]


def test_edf_info_rejects_hypnogram():
    """Prueba el rechazo explicativo al subir un hipnograma en vez de un PSG."""
    if not SAMPLE_HYP.exists():
        print("Aviso: Archivo de prueba Hypnogram no encontrado, omitiendo test")
        return

    with open(SAMPLE_HYP, "rb") as f:
        response = client.post(
            "/api/v1/edf/info",
            files={"file": (SAMPLE_HYP.name, f, "application/octet-stream")},
        )

    assert response.status_code == 400
    data = response.json()
    assert "hipnograma" in data["detail"].lower()


def test_analyze_window_multimodel():
    """Prueba el análisis de una ventana de 30s con selección múltiple (SVM + LightGBM)."""
    if not SAMPLE_PSG.exists():
        print("Aviso: Archivo de prueba PSG no encontrado, omitiendo test")
        return

    with open(SAMPLE_PSG, "rb") as f:
        response = client.post(
            "/api/v1/analyze/window",
            files={"file": (SAMPLE_PSG.name, f, "application/octet-stream")},
            data={"window_index": 0, "models": ["svm", "lightgbm"]},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["window_index"] == 0
    assert data["start_second"] == 0.0
    assert data["end_second"] == 30.0
    assert "svm" in data["models_analyzed"]
    assert "lightgbm" in data["models_analyzed"]

    # Verificación predicción SVM
    pred_svm = data["predictions"]["svm"]
    assert pred_svm["stage"] in ["Wake", "N1", "N2", "N3", "REM"]
    assert 0.0 <= pred_svm["confidence"] <= 1.0
    assert len(pred_svm["probabilities"]) == 5

    # Verificación predicción LightGBM
    pred_lgb = data["predictions"]["lightgbm"]
    assert pred_lgb["stage"] in ["Wake", "N1", "N2", "N3", "REM"]
    assert 0.0 <= pred_lgb["confidence"] <= 1.0
    assert len(pred_lgb["probabilities"]) == 5

    # Verificación de consenso
    assert "consensus" in data
    assert data["consensus"]["consensus_stage"] in ["Wake", "N1", "N2", "N3", "REM"]
    assert isinstance(data["consensus"]["agreement"], bool)
    assert 0.0 <= data["consensus"]["agreement_percentage"] <= 100.0


def test_analyze_batch():
    """Prueba el análisis por lotes de 2 épocas consecutivas."""
    if not SAMPLE_PSG.exists():
        print("Aviso: Archivo de prueba PSG no encontrado, omitiendo test")
        return

    with open(SAMPLE_PSG, "rb") as f:
        response = client.post(
            "/api/v1/analyze/batch",
            files={"file": (SAMPLE_PSG.name, f, "application/octet-stream")},
            data={"start_window": 0, "max_windows": 2, "models": ["svm", "lightgbm"]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["total_windows_analyzed"] == 2
    assert len(data["epochs"]) == 2
    assert "svm" in data["stage_distribution_by_model"]
    assert "lightgbm" in data["stage_distribution_by_model"]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Ejecutando tests directos...")
    test_root_endpoint()
    print("[OK] test_root_endpoint")
    test_health_endpoint()
    print("[OK] test_health_endpoint")
    test_models_catalog_endpoint()
    print("[OK] test_models_catalog_endpoint")
    test_edf_info_endpoint()
    print("[OK] test_edf_info_endpoint")
    test_edf_info_rejects_hypnogram()
    print("[OK] test_edf_info_rejects_hypnogram")
    test_analyze_window_multimodel()
    print("[OK] test_analyze_window_multimodel")
    test_analyze_batch()
    print("[OK] test_analyze_batch")
    print("\n>>> TODOS LOS TESTS DE LA API PASARON EXITOSAMENTE! <<<")
