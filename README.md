# Monitor Clínico de Polisomnografía AI — MAIA Grupo 25

**Efecto del contexto temporal en la clasificación automática de estadios del sueño sobre señales crudas EEG/EOG**  
*Maestría en Inteligencia Artificial · Universidad de los Andes · Bogotá, Colombia*

---

## 1. Enlaces Relevantes y Despliegue en Producción

| Recurso | Enlace Oficial | Descripción |
| :--- | :--- | :--- |
| **Repositorio GitHub** | https://github.com/amquinteroc1/MAIA-Grupo25-Proyecto | Código fuente, notebooks, modelos finales, datos y pruebas |
| **Aplicación Web (`/ui`)** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/ui | Dashboard clínico stateless con Wizard instructivo, visor PSG 60fps, mapa de estadios y matriz 3×3 |
| **Swagger OpenAPI (`/docs`)** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/docs | Documentación interactiva y pruebas REST |
| **ReDoc (`/redoc`)** | https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/redoc | Especificación técnica de endpoints y esquemas |
| **MLflow Tracking** | https://sleep-mlflow.whitesky-80270a35.eastus.azurecontainerapps.io | Experimento `sleep-edfx-tesis-E0-E1-E2` (9 corridas, métricas y figuras Wilcoxon) |
| **Hugging Face Data** | https://huggingface.co/datasets/maia-grupo25/sleep-edfx-data | Tensores procesados (`SC`/`ST`), splits por sujeto y metadatos |
| **Hugging Face Models** | https://huggingface.co/datasets/maia-grupo25/sleep-edfx-models | Los 9 checkpoints oficiales (`E0`, `E1`, `E2` × semillas `42`, `123`, `2026`) |
| **Dataset Fuente** | https://physionet.org/content/sleep-edfx/1.0.0/ | PhysioNet Sleep-EDF Expanded (`197` registros, `100` sujetos) |

---

## 2. Descripción del Proyecto y Resultados

Sistema clínico-investigativo de apoyo al diagnóstico automático de las 5 etapas canónicas del sueño (**Wake, N1, N2, N3 y REM**, estándar **R&K / AASM**) a partir de 3 canales crudos a 100 Hz (`EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`) en ventanas de 30 s. Se evalúan tres arquitecturas bajo partición estricta por sujeto (`SC`: 49 train / 13 val / 16 test; `ST` externo: 22 sujetos) y normalización Z-score ajustada en `SC-train` (`Datos/norm_params.npz`), entrenadas con 3 semillas (`42`, `123`, `2026` — **9 corridas**):

| Modelo | Contexto Temporal | Accuracy `SC-test` | F1-Macro `SC-test` | Accuracy `ST` (externo) | F1-Macro `ST` | F1 `N1` (`ST`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **E0** (`SleepCNN1D`) | 1 época (30 s) | 78,36 % | 0,7601 | 65,26 % | 0,6090 | 0,3603 |
| **E1** (`SleepCNN1D_E1`) | 5 épocas · Concatenación | 81,51 % | 0,7814 | 61,19 % | 0,5893 | 0,3282 |
| **E2** (`SleepCNN1D_E2`) ⭐ | 5 épocas · **GRU (256 u.)** | **82,34 %** | **0,7888** | **67,81 %** | **0,6354** | **0,3884** |

- **Conclusión experimental:** **E2 (CNN 1D + GRU)** alcanza la mejor precisión y F1-Macro en dominio interno (`SC-test`: 39.961 épocas) y generalización externa (`ST`: 42.003 épocas, retención cross-domain $R = +0.92$).

---

## 3. Estructura del Repositorio y Carpetas Obligatorias

| Carpeta / Archivo | Contenido |
| :--- | :--- |
| **`Notebooks/`** | Cuadernos reproducibles de entrenamiento y análisis (`train_e0.ipynb`, `train_e1.ipynb`, `train_e2.ipynb`) |
| **`Modelos/`** | Checkpoints finales entrenados en PyTorch (`best_model_E0_seed42.pt`, `best_model_E1_seed42.pt`, `best_model_E2_seed42.pt`) + guía de descarga de las 9 semillas |
| **`Datos/`** | Estructura de señales, inventario EDA (`sleep_edf_records.csv`, `sleep_edf_summary.json`) y parámetros Z-score oficiales (`norm_params.npz`) |
| `src/` | Código modular: `config.py`, `data_tesis/`, `models/`, `training/`, `evaluation/`, `inference/predict.py`, `api/` (`main.py`, `service.py`, `schemas.py`, `static/index.html`) y `utils/` |
| `kaggle/` | Kernels batch asíncronos (`e0`, `e1`, `e2`) para auditoría remota |
| `tests/test_api.py` | Pruebas automatizadas E2E sobre API, `/ui`, modelos y mapa de estadios |
| `Dockerfile.api`, `Dockerfile.mlflow`, `docker-compose.yml` | Empaquetado reproducible para local y Azure Container Apps |

---

## 4. Dependencias y Entorno de Ejecución

- **Runtime:** Python `3.11+` (probado también en `3.14` local y `python:3.11-slim` en Docker).
- **Dependencias clave** ([`requirements.txt`](requirements.txt) + PyTorch CPU/CUDA): `torch`, `fastapi`, `uvicorn`, `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `mlflow`, `huggingface_hub`, `python-dotenv`, `kaggle`, `httpx`, `dvc[s3]`.

```powershell
# Entorno virtual en Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # o wheel CUDA
pip install -r requirements.txt
```

---

## 5. Parametrización y Credenciales de Ejemplo

El comportamiento del sistema se parametriza sin editar código mediante:
1. **`src/config.py`:** hiperparámetros científicos compartidos (arquitectura CNN/GRU, `seq_len=5`, semillas `[42, 123, 2026]`, `batch_size=128`, `lr=1e-3`, `early_stopping_patience=10`).
2. **Variables de entorno (`.env`):** plantilla documentada en [`.env.example`](.env.example).

```dotenv
# Ejemplo de configuración (.env) — NO versionar credenciales reales
HUGGING_FACE_API_KEY=hf_tu_token_ejemplo_aqui
HF_DATA_REPO=maia-grupo25/sleep-edfx-data
HF_MODEL_REPO=maia-grupo25/sleep-edfx-models
HF_REPO_TYPE=dataset

MLFLOW_TRACKING_URI=http://localhost:5000
API_PORT=8001
API_URL=http://localhost:8001

# Auditoría batch remota opcional (Kaggle API)
KAGGLE_USERNAME=usuario_ejemplo
KAGGLE_KEY=KGAT_token_ejemplo_aqui
KAGGLE_KERNEL_E0=sleep-edfx-e0-batch
KAGGLE_KERNEL_E1=sleep-edfx-e1-batch
KAGGLE_KERNEL_E2=sleep-edfx-e2-batch
```

| Variable | Propósito | Valor por defecto |
| :--- | :--- | :--- |
| `API_PORT` | Puerto expuesto en host para FastAPI (`/ui`, `/docs`) | `8001` |
| `HUGGING_FACE_API_KEY` | Token para descarga/sincronización de modelos desde HF Hub | *(opcional si repo público)* |
| `HF_DATA_REPO` / `HF_MODEL_REPO` | Repositorios oficiales en Hugging Face | `maia-grupo25/sleep-edfx-{data,models}` |
| `MLFLOW_TRACKING_URI` | URI del servidor de tracking MLflow | `http://localhost:5000` |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Credenciales para auditoría remota por lotes | *(opcional)* |

---

## 6. Pasos de Despliegue (Local y Nube Azure)

### Despliegue Local con Docker Compose
```powershell
Copy-Item .env.example .env
docker compose up -d --build
```
- **Dashboard Web:** http://localhost:8001/ui · **Swagger:** http://localhost:8001/docs · **MLflow:** http://localhost:5000

### Ejecución Local Directa (sin Docker)
```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload
```

### Despliegue en Producción (Azure Container Apps + GitHub Container Registry)
```powershell
# 1. Construir y publicar imágenes autocontenidas en GHCR
docker build -f Dockerfile.api -t ghcr.io/amquinteroc1/sleep-ai-api:latest .
docker build -f Dockerfile.mlflow -t ghcr.io/amquinteroc1/sleep-mlflow:latest .
docker push ghcr.io/amquinteroc1/sleep-ai-api:latest
docker push ghcr.io/amquinteroc1/sleep-mlflow:latest

# 2. Crear grupo de recursos, entorno serverless (Consumption) y desplegar contenedores
az group create --name rg-sleep-ai --location eastus
az containerapp env create --name env-sleep-ai --resource-group rg-sleep-ai --location eastus

az containerapp create --name sleep-ai-api --resource-group rg-sleep-ai --environment env-sleep-ai `
  --image ghcr.io/amquinteroc1/sleep-ai-api:latest --target-port 8000 --ingress external `
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1 `
  --secrets hf-token="<HF_TOKEN>" kaggle-key="<KAGGLE_KEY>" `
  --env-vars HUGGING_FACE_API_KEY=secretref:hf-token KAGGLE_USERNAME="<USER>" KAGGLE_KEY=secretref:kaggle-key

az containerapp create --name sleep-mlflow --resource-group rg-sleep-ai --environment env-sleep-ai `
  --image ghcr.io/amquinteroc1/sleep-mlflow:latest --target-port 5000 --ingress external `
  --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1
```

---

## 7. Ejemplos de Uso

### A. Uso Interactivo en el Dashboard Web (`/ui`)
1. Al ingresar por primera vez se abre el **Wizard instructivo** de 4 pasos (ocultable con *"No volver a mostrar"* y reabrible desde **❓ Guía rápida**).
2. Arrastra un estudio `*-PSG.edf` (ej. `SC4001E0-PSG.edf`) en el panel izquierdo (`preload=False`, carga única).
3. Explora el **Mapa Visual de Estadios por colores** debajo de la línea de tiempo y haz clic en las etiquetas (`Wake`, `N1`, `N2`, `N3`, `REM`) para saltar directo a cualquier fase del sueño con actualización automática del diagnóstico clínico.
4. Cambia entre **Consenso Tesis (E0+E1+E2)**, arquitectura individual, **Todas las 9 corridas** o **Matriz 3×3 personalizada**.

### B. Ejemplos de Consumo REST API
```bash
# 1. Verificar salud y listar modelos disponibles (9 corridas E0-E2)
curl https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/health
curl https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/api/v1/models

# 2. Registrar archivo PSG una sola vez (retorna edf_id)
curl -X POST https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/api/v1/edf/upload \
  -F "file=@data/sleep-cassette/SC4001E0-PSG.edf"

# 3. Obtener mapa rápido de estadios de toda la noche y analizar ventana #1000
curl https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/api/v1/edf/<EDF_ID>/timeline
curl -X POST https://sleep-ai-api.whitesky-80270a35.eastus.azurecontainerapps.io/api/v1/edf/<EDF_ID>/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_index": 1000, "models": ["e0_seed42", "e1_seed42", "e2_seed42"]}'
```

### C. Pruebas Automatizadas y Reproducibilidad Científica
```powershell
pytest tests/test_api.py -v                         # Suite de integración (5/5 pruebas)
python -m src.data_tesis.test_data_tesis            # Integridad de tensores y particiones por sujeto
python -m src.utils.register_thesis_mlflow          # Registro de las 9 corridas en MLflow
```

---

## 8. Integrantes — MAIA Grupo 25

**Maestría en Inteligencia Artificial · Universidad de los Andes (Bogotá, Colombia)**

- **Diego Andrés Burbano Arturo** — `d.burbanoa@uniandes.edu.co`
- **Luigi Daniele Di Biase Giusti** — `l.dib@uniandes.edu.co`
- **Omar Darío Díaz Hernandez** — `o.diazh@uniandes.edu.co`
- **Andrés Mauricio Quintero Correal** — `am.quinteroc1@uniandes.edu.co`
