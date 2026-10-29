# Guía de Almacenamiento: Hugging Face Hub & DVC

El proyecto utiliza **Hugging Face Hub** como repositorio oficial de almacenamiento en la nube para datasets y modelos, eliminando la dependencia de credenciales temporales o caducas de AWS Academy S3.

Repositorios oficiales de la organización:
- 📦 **Datasets:** `maia-grupo25/sleep-edfx-data`
- 🧠 **Modelos:** `maia-grupo25/sleep-edfx-models`

---

## 1. Instalación de Dependencias

Instala el SDK de Hugging Face y soporte para variables `.env`:

```powershell
pip install huggingface_hub python-dotenv
```

*(O instala todo el entorno del proyecto con `pip install -r requirements.txt`).*

---

## 2. Autenticación con Variables de Entorno

Se requiere un token de acceso personal con permisos de **Write** generado en:
👉 [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (su formato inicia con `hf_...`).

### Opción A: Archivo `.env` (Recomendado)
Copia la plantilla y edita tu archivo `.env`:

```powershell
Copy-Item .env.example .env
```

Contenido en `.env`:
```env
HUGGING_FACE_API_KEY=hf_tu_token_aqui
HF_DATA_REPO=maia-grupo25/sleep-edfx-data
HF_MODEL_REPO=maia-grupo25/sleep-edfx-models
HF_REPO_TYPE=dataset
```

> **Seguridad:** El archivo `.env` está en `.gitignore` y **nunca** debe comitearse al repositorio Git.

### Opción B: Variable en Terminal PowerShell (Sesión Actual)
```powershell
$env:HUGGING_FACE_API_KEY = "hf_tu_token_aqui"
```

### Opción C: Permanente en Windows (Usuario)
```powershell
[System.Environment]::SetEnvironmentVariable('HUGGING_FACE_API_KEY', 'hf_tu_token_aqui', 'User')
```

### Opción D: Linux / macOS / Bash
```bash
export HUGGING_FACE_API_KEY="hf_tu_token_aqui"
```

---

## 3. Verificar Autenticación

Ejecuta el verificador integrado para validar credenciales y conexión a ambos repositorios:

```powershell
python -m src.utils.hf_hub verify
```

Salida esperada:
```text
✓ Autenticado exitosamente como usuario: '<tu_usuario>'
✓ Repositorio 'maia-grupo25/sleep-edfx-models' (dataset) accesible.
✓ Repositorio 'maia-grupo25/sleep-edfx-data' (dataset) accesible.
```

---

## 4. Uso Rápido: Descarga y Sincronización

### Descargar modelos y datos del proyecto:
```powershell
# Descarga los modelos (SVM, LightGBM, CNN 1D) si faltan en disco
python -m src.utils.hf_hub pull-models

# Descarga el dataset oficial
python -m src.utils.hf_hub pull-data
```

### Entrenar modelos y versionar automáticamente en Hugging Face + DVC:
```powershell
# CNN 1D con Deep Learning
python -m src.models.train_cnn --push-hf

# LightGBM Ensemble
python -m experiments.experiment_lightgbm --push-hf

# SVM Baseline
python -m experiments.experiment_svm --push-hf
```

### Sincronizar todos los modelos locales existentes:
```powershell
python -m src.utils.hf_hub push-models
```

---

## 5. ¿Cómo interactúan Hugging Face y DVC?

1. **Git:** Almacena el código y los archivos puntero `.dvc` (ejemplo: `models/svm_sleep.joblib.dvc`).
2. **DVC:** Gestiona el linaje de datos y genera los metadatos de integridad (`dvc add`).
3. **Hugging Face Hub:** Almacena los archivos pesados reales con historial de versiones, tags y visualización web para el equipo.
