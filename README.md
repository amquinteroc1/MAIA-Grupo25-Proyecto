# Monitor Clínico de Polisomnografía AI — MAIA Grupo 25

**Efecto del contexto temporal en la clasificación automática de estadios del sueño sobre señales crudas EEG/EOG**  
*Maestría en Inteligencia Artificial · Universidad de los Andes · Bogotá, Colombia*

---

## Despliegue en Producción (Azure Container Apps)

| Servicio | URL Pública (HTTPS) | Descripción |
| :--- | :--- | :--- |
| **Dashboard Clínico Web** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/ui | Interfaz stateless responsiva (HTML5 + Canvas 60fps), visor PSG, mapa de colores de estadios y matriz 3×3 de modelos |
| **API Swagger OpenAPI** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/docs | Documentación interactiva y pruebas REST |
| **Especificación ReDoc** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/redoc | Referencia técnica de esquemas y endpoints |
| **MLflow Tracking** | https://sleep-mlflow.whitesky-80270a35.eastus.azurecontainerapps.io | Experimento `sleep-edfx-tesis-E0-E1-E2` (9 corridas, métricas y figuras Wilcoxon) |

---

## 1. Resumen Científico y Resultados

Se comparan tres arquitecturas profundas sobre **[Sleep-EDF Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/)** usando 3 canales crudos a 100 Hz (`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`), épocas de 30 s (estándar **R&K / AASM**: `Wake`, `N1`, `N2`, `N3`, `REM`), partición estricta por sujeto (`SC`: 49 train / 13 val / 16 test; `ST` externo: 22 sujetos) y normalización Z-score ajustada en `SC-train` (`norm_params.npz`). Cada arquitectura se entrenó con 3 semillas (`42`, `123`, `2026` — **9 corridas independientes**):

| Modelo | Contexto Temporal | Accuracy `SC-test` | F1-Macro `SC-test` | Accuracy `ST` (externo) | F1-Macro `ST` | F1 `N1` (`ST`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **E0** (`SleepCNN1D`) | 1 época (30 s) | 78,36 % | 0,7601 | 65,26 % | 0,6090 | 0,3603 |
| **E1** (`SleepCNN1D_E1`) | 5 épocas · Concatenación | 81,51 % | 0,7814 | 61,19 % | 0,5893 | 0,3282 |
| **E2** (`SleepCNN1D_E2`) ⭐ | 5 épocas · **GRU (256 u.)** | **82,34 %** | **0,7888** | **67,81 %** | **0,6354** | **0,3884** |

- **Hallazgo principal:** **E2 (CNN 1D + GRU)** logra el mejor desempeño tanto en dominio interno (`SC-test`: 39.961 épocas) como en generalización externa cross-domain (`ST`: 42.003 épocas, retención $R = +0.92$).
- **Artefactos oficiales en Hugging Face:** [`maia-grupo25/sleep-edfx-data`](https://huggingface.co/datasets/maia-grupo25/sleep-edfx-data) y [`maia-grupo25/sleep-edfx-models`](https://huggingface.co/datasets/maia-grupo25/sleep-edfx-models).

---

## 2. Arquitectura del Producto

- **Frontend Web Stateless (`/ui`) + API REST (FastAPI):** un único contenedor ligero (`Dockerfile.api`) que cachea archivos `*-PSG.edf` con `preload=False` (`< 5 MB` RAM), grafica ondas en `<canvas>` a 60 fps, escanea el mapa visual de estadios de toda la noche en `< 1 s` (`/api/v1/edf/{id}/timeline`) y ejecuta inferencia reactiva al deslizar la línea de tiempo (~150 ms) con soporte para las **9 corridas oficiales (`Matriz 3×3`)**.
- **Auditoría Batch Remota (Kaggle API):** procesamiento asíncrono desacoplado (`kaggle/e{0,1,2}/`) para validar checkpoints y consolidar métricas de cohorte completa (`SC-test`) sin sobrecargar el servidor web.
- **MLflow Tracking (`Dockerfile.mlflow`):** servidor centralizado con `--workers 1` y el experimento consolidado `sleep-edfx-tesis-E0-E1-E2`.

---

## 3. Inicio Rápido

### Opción A — Docker Compose (Stack completo: API + `/ui` + MLflow)
```powershell
Copy-Item .env.example .env   # Configurar tokens de Hugging Face y Kaggle
docker compose up -d --build
```
- **Dashboard Web:** http://localhost:8001/ui
- **Swagger OpenAPI:** http://localhost:8001/docs
- **MLflow UI:** http://localhost:5000

### Opción B — Ejecución local en Python (`.venv`)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload
```

### Pruebas y Reproducibilidad
```powershell
# Suite de integración (API, /ui, 9 modelos, mapa de estadios y Kaggle config)
pytest tests/test_api.py -v

# Reconstrucción de tensores, validación y entrenamiento (9 corridas)
python -m src.data_tesis.crear_dataset_tesis --fase todo
python -m src.data_tesis.test_data_tesis
python -m src.training.train --experiment all --seed all

# Evaluación externa ST, matrices de confusión y registro en MLflow
python -m src.evaluation.evaluacion_st
python -m src.evaluation.generar_matrices_confusion
python -m src.utils.register_thesis_mlflow
```

---

## 4. Estructura del Repositorio

```text
├── Dockerfile.api / Dockerfile.mlflow / docker-compose.yml
├── src/
│   ├── config.py                    # Hiperparámetros y protocolo compartido
│   ├── data_tesis/                  # Construcción y validación de tensores SC/ST
│   ├── models/                      # Arquitecturas E0 (cnn1d), E1 (concat), E2 (GRU)
│   ├── training/                    # Pipeline de entrenamiento reproducible (3 semillas)
│   ├── evaluation/                  # Evaluación ST, matrices de confusión y test de Wilcoxon
│   ├── inference/predict.py         # Motor de inferencia unificado (9 corridas E0-E2)
│   ├── api/                         # FastAPI (main, service, schemas) + frontend stateless (static/)
│   └── utils/                       # Integraciones Hugging Face, Kaggle Batch API y MLflow
├── kaggle/                          # Kernels batch remotos E0, E1, E2
├── notebooks/                       # Notebooks reproducibles de entrenamiento (train_e0..e2)
├── outputs/                         # Resúmenes consolidados de auditoría y pruebas de Wilcoxon
├── tests/test_api.py                # Pruebas automatizadas E2E
└── docs/                            # Documentación de datos y versionado DVC/HF
```

---

## 5. Equipo de Desarrollo

**MAIA Grupo 25 — Maestría en Inteligencia Artificial · Universidad de los Andes (Bogotá, Colombia)**

- **Diego Andrés Burbano Arturo** — `d.burbanoa@uniandes.edu.co`
- **Luigi Daniele Di Biase Giusti** — `l.dib@uniandes.edu.co`
- **Omar Darío Díaz Hernandez** — `o.diazh@uniandes.edu.co`
- **Andrés Mauricio Quintero Correal** — `am.quinteroc1@uniandes.edu.co`
