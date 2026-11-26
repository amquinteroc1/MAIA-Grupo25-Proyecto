"""
test_api.py
Suite de pruebas de integración para la API REST y Dashboard Web Stateless (/ui)
de Clasificación de Etapas del Sueño (MAIA Grupo 25 — E0, E1, E2 × 3 semillas).
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

SAMPLE_PSG = ROOT / "data" / "sleep-cassette" / "SC4001E0-PSG.edf"
SAMPLE_HYP = ROOT / "data" / "sleep-cassette" / "SC4001EC-Hypnogram.edf"

ALL_9_THESIS = [
    f"{arch}_seed{seed}"
    for arch in ("e0", "e1", "e2")
    for seed in ("42", "123", "2026")
]

client = TestClient(app)


def test_root_and_ui_endpoints():
    """Verifica el endpoint raíz /, el dashboard stateless /ui y los assets institucionales."""
    r_root = client.get("/")
    assert r_root.status_code == 200
    data = r_root.json()
    assert data["status"] == "online"
    assert data["web_dashboard"] == "/ui"
    assert data["swagger_docs"] == "/docs"

    r_ui = client.get("/ui")
    assert r_ui.status_code == 200
    assert "text/html" in r_ui.headers.get("content-type", "")
    html = r_ui.text
    for marker in ("timelineCanvas", "stageJumpBar", "matrix-3x3", "uniandes-logo.png", "site-footer"):
        assert marker in html

    assert client.get("/favicon.ico").status_code == 200
    assert client.get("/static/assets/uniandes-logo.png").status_code == 200


def test_health_and_models_catalog():
    """Verifica /health y que las 9 corridas oficiales E0/E1/E2 estén disponibles."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    health = r_health.json()
    assert health["status"] == "online"

    r_models = client.get("/api/v1/models")
    assert r_models.status_code == 200
    catalog = {m["id"]: m for m in r_models.json()["models"]}
    for mid in ["e0", "e1", "e2"] + ALL_9_THESIS:
        assert mid in catalog, f"Modelo faltante en catálogo: {mid}"
        assert catalog[mid]["available"] is True, f"Modelo no disponible: {mid}"


def test_edf_info_rejects_hypnogram():
    """Verifica el rechazo explicativo al subir un hipnograma en vez de un PSG."""
    if not SAMPLE_HYP.exists():
        return
    with SAMPLE_HYP.open("rb") as f:
        response = client.post(
            "/api/v1/edf/info",
            files={"file": (SAMPLE_HYP.name, f, "application/octet-stream")},
        )
    assert response.status_code == 400
    assert "hipnograma" in response.json()["detail"].lower()


def test_cached_edf_signals_timeline_and_inference():
    """Verifica el flujo completo: upload cacheado -> señales -> mapa de estadios -> inferencia E0/E1/E2."""
    if not SAMPLE_PSG.exists():
        return

    with SAMPLE_PSG.open("rb") as f:
        r_up = client.post(
            "/api/v1/edf/upload",
            files={"file": (SAMPLE_PSG.name, f, "application/octet-stream")},
        )
    assert r_up.status_code == 200
    up = r_up.json()
    edf_id = up["edf_id"]
    assert up["is_valid"] is True
    assert up["total_windows_30s"] > 0

    # Señales para graficar
    r_sig = client.get(f"/api/v1/edf/{edf_id}/window/500/signals")
    assert r_sig.status_code == 200
    sig = r_sig.json()
    assert set(sig["channels"].keys()) == {"EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"}

    # Mapa de estadios de toda la noche
    r_tl = client.get(f"/api/v1/edf/{edf_id}/timeline?max_points=120")
    assert r_tl.status_code == 200
    tl = r_tl.json()
    assert len(tl["points"]) >= 100
    assert sum(tl["stage_counts"].values()) == len(tl["points"])

    # Inferencia multimodelo con consenso (E0 + E1 + E2 y combinación personalizada)
    r_inf = client.post(
        f"/api/v1/edf/{edf_id}/analyze",
        json={"window_index": 500, "models": ["e0_seed42", "e1_seed123", "e2_seed2026"]},
    )
    assert r_inf.status_code == 200
    inf = r_inf.json()
    assert len(inf["predictions"]) == 3
    assert inf["consensus"]["consensus_stage"] in ["Wake", "N1", "N2", "N3", "REM"]
    assert 0.0 <= inf["consensus"]["agreement_percentage"] <= 100.0


def test_kaggle_config_endpoint():
    """Verifica el endpoint de configuración del módulo de auditoría batch remota."""
    r = client.get("/api/v1/kaggle/config")
    assert r.status_code == 200
    data = r.json()
    assert "configured" in data
    assert set(data["kernels"].keys()) == {"e0", "e1", "e2"}
