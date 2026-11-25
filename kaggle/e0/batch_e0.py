"""
Runner batch para Kaggle Kernels (E0 - CNN 1D sin contexto).
Descarga el checkpoint oficial y los resúmenes de las 3 semillas desde
Hugging Face Hub (maia-grupo25) y expone las métricas y validación en /kaggle/working/.
"""
import json
import os
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

EXPERIMENT = "E0"
SEEDS = [42, 123, 2026]
DATA_REPO = os.getenv("HF_DATA_REPO", "maia-grupo25/sleep-edfx-data")
MODEL_REPO = os.getenv("HF_MODEL_REPO", "maia-grupo25/sleep-edfx-models")
HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY") or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HUGGING_FACE_API_KEY")
    except Exception:
        HF_TOKEN = None

OUT_DIR = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = {"experiment": EXPERIMENT, "source": f"hf://{DATA_REPO}", "seeds": {}, "status": "ok"}

ckpt_path = Path(
    hf_hub_download(
        repo_id=MODEL_REPO,
        repo_type="dataset",
        filename=f"tesis-final-2026/experiments/{EXPERIMENT}/seed42/best_model_{EXPERIMENT}_seed42.pt",
        token=HF_TOKEN,
    )
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
summary["checkpoint_verified"] = {
    "file": ckpt_path.name,
    "size_mb": round(ckpt_path.stat().st_size / 1e6, 2),
    "epoch": ckpt.get("epoch"),
    "val_f1_macro": ckpt.get("val_f1_macro"),
}

for seed in SEEDS:
    p = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=f"tesis-final-2026/experiments/{EXPERIMENT}/seed{seed}/run_summary.json",
        token=HF_TOKEN,
    )
    summary["seeds"][str(seed)] = json.loads(Path(p).read_text(encoding="utf-8"))

out_file = OUT_DIR / f"batch_summary_{EXPERIMENT}.json"
out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Batch {EXPERIMENT} completado OK -> {out_file}")
