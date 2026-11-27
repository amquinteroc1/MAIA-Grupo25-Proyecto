# Modelos/ — Artefactos del Modelo Final Guardado (PyTorch)

Esta carpeta contiene los checkpoints oficiales representativos (`.pt`) de las tres arquitecturas profundas comparadas en la tesis de **MAIA Grupo 25 — Universidad de los Andes**:

| Archivo | Arquitectura | Contexto Temporal | Accuracy `SC-test` (Prom. 3 seeds) | F1-Macro `SC-test` | F1-Macro `ST` (Externo) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `best_model_E0_seed42.pt` | `SleepCNN1D` (`src/models/cnn1d.py`) | 1 época (30 s) | 78,36 % | 0,7601 | 0,6090 |
| `best_model_E1_seed42.pt` | `SleepCNN1D_E1` (`src/models/cnn1d_e1.py`) | 5 épocas · Concatenación | 81,51 % | 0,7814 | 0,5893 |
| `best_model_E2_seed42.pt` ⭐ | `SleepCNN1D_E2` (`src/models/cnn1d_e2.py`) | 5 épocas · **GRU (256 u.)** | **82,34 %** | **0,7888** | **0,6354** |

> **Modelo recomendado en producción:** `best_model_E2_seed42.pt` (`E2`, CNN 1D + GRU), con retención cross-domain $R = +0.92$ en el conjunto externo Sleep-Telemetry (`ST`).

## Descarga de las 9 Corridas Completas (Semillas `42`, `123`, `2026`)

Para mantener el repositorio ágil, aquí se versionan los checkpoints principales (`seed42`). Las **9 corridas oficiales completas** (`E0`, `E1`, `E2` × semillas `42`, `123`, `2026`) están alojadas en Hugging Face Hub ([`maia-grupo25/sleep-edfx-models`](https://huggingface.co/datasets/maia-grupo25/sleep-edfx-models)) y se descargan automáticamente en tiempo de ejecución o mediante:

```powershell
python -m src.utils.hf_hub pull-models
```
