# Pipeline de datos — Proyecto de Grado
## Clasificación de estadios del sueño · Grupo 25

---

## 1. Resumen

El pipeline de datos implementa el **protocolo experimental unificado** (Informe de Avance, sección 2) y genera los tensores listos para entrenar y evaluar las condiciones E0, E1 y E2. Todos los artefactos están en `data/processed/` y se suben a Hugging Face (`maia-grupo25/sleep-edfx-data`).

**Branch:** `cnn-psg-tesis`  
**Script principal:** `src/data_tesis/crear_dataset_tesis.py`  
**Validación:** `src/data_tesis/test_data_tesis.py` — 21/21 checks pasados

---

## 2. Conjuntos de datos

### 2.1 Particionamiento

| Conjunto | Origen | Sujetos | Uso | Modelos se entrenan aquí | Se ajustan hiperparámetros |
|----------|--------|---------|-----|:------------------------:|:--------------------------:|
| **SC-train** | Sleep Cassette | 97 (64%) | Entrenamiento + norm params | ✅ | ❌ |
| **SC-val** | Sleep Cassette | 25 (16%) | Early stopping, selección de hiperparámetros | ❌ | ✅ |
| **SC-test** | Sleep Cassette | 31 (20%) | Evaluación final en dominio | ❌ | ❌ |
| **ST** | Sleep Telemetry | 44 (100%) | Evaluación fuera de dominio | ❌ | ❌ |

- Split por sujeto con `GroupShuffleSplit`, seed=42.
- Ningún sujeto aparece en más de una partición.
- SC-test y ST están **sellados** — no se usan para ninguna decisión hasta el reporte final.

### 2.2 Épocas por conjunto

#### E0 — por época (3, 3000)

| Conjunto | Total | Wake | N1 | N2 | N3 | REM |
|----------|------:|-----:|---:|---:|---:|----:|
| SC-train | 126,157 | 43,152 (34.2%) | 14,267 (11.3%) | 45,470 (36.0%) | 7,011 (5.6%) | 16,257 (12.9%) |
| SC-val | 30,785 | 10,340 | 3,207 | 10,181 | 2,786 | 4,271 |
| SC-test | 38,537 | 12,459 | 4,048 | 13,481 | 3,242 | 5,307 |
| ST | 42,471 | 4,203 | 3,653 | 19,851 | 6,415 | 8,349 |

#### Secuencias E1/E2 — (5, 3, 3000)

| Conjunto | Secuencias | Épocas E0 | Excluidas por borde |
|----------|----------:|-----------:|--------------------:|
| SC-train | 125,769 | 126,157 | 388 |
| SC-val | 30,685 | 30,785 | 100 |
| SC-test | 38,413 | 38,537 | 124 |
| ST | 42,295 | 42,471 | 176 |

Las épocas de borde (2 inicio + 2 final por registro) se excluyen — no se padean.

---

## 3. Estructura de archivos

```
data/processed/
├── e0/                               # Tensores por época
│   ├── X_train.npy                   # (126157, 3, 3000) float32
│   ├── y_train.npy                   # (126157,) int64
│   ├── groups_train.npy              # (126157,) int64 — sujeto por época
│   ├── X_val.npy                     # (30785, 3, 3000)
│   ├── y_val.npy
│   ├── groups_val.npy
│   ├── X_test.npy                    # (38537, 3, 3000)
│   ├── y_test.npy
│   ├── groups_test.npy
│   ├── X_st.npy                      # (42471, 3, 3000)
│   ├── y_st.npy
│   └── groups_st.npy
├── seq/                              # Secuencias de 5 épocas
│   ├── X_seq_train.npy               # (125769, 5, 3, 3000) float32
│   ├── y_seq_train.npy               # (125769,) int64 — etiqueta de la época central
│   ├── groups_seq_train.npy          # (125769,) int64
│   ├── eligible_indices_train.npy    # (125769,) int64 — índices en E0 de las épocas centrales
│   ├── X_seq_val.npy                 # (30685, 5, 3, 3000)
│   ├── ...                           # (mismos archivos para val, test, st)
│   └── eligible_indices_st.npy
└── meta/                             # Trazabilidad
    ├── norm_params.npz               # media y std por canal (estimados solo con SC-train)
    ├── inventory_records.csv         # 197 registros con exclusiones y hashes MD5 de EDF
    └── split_info.json               # sujetos, conteos, proporciones, exclusiones de borde
```

---

## 4. Cómo cargar los datos

### E0 — para entrenar la línea base (Omar)

```python
import numpy as np

X_train = np.load("data/processed/e0/X_train.npy")     # (126157, 3, 3000)
y_train = np.load("data/processed/e0/y_train.npy")      # (126157,)
X_val   = np.load("data/processed/e0/X_val.npy")        # (30785, 3, 3000)
y_val   = np.load("data/processed/e0/y_val.npy")        # (30785,)

# Para PyTorch
import torch
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
```

### E1/E2 — secuencias con contexto temporal

```python
X_seq_train = np.load("data/processed/seq/X_seq_train.npy")   # (125769, 5, 3, 3000)
y_seq_train = np.load("data/processed/seq/y_seq_train.npy")   # (125769,) — época central
```

### Para Google Colab (descarga desde HF)

```python
# Opción 1: huggingface-cli
!pip install huggingface_hub
!huggingface-cli download maia-grupo25/sleep-edfx-data --repo-type dataset --local-dir data/processed/

# Opción 2: programático
from huggingface_hub import snapshot_download
snapshot_download(repo_id="maia-grupo25/sleep-edfx-data", repo_type="dataset", local_dir="data/processed/")
```

---

## 5. Reglas críticas para Omar y Luigi

### ⚠️ NO hacer

1. **No volver a normalizar.** Los datos ya están normalizados con los parámetros de SC-train. Normalizar de nuevo distorsiona los valores.
2. **No usar SC-test ni ST para ajustar hiperparámetros** — ni para early stopping, ni para seleccionar arquitectura. Solo SC-val.
3. **No descartar ejecuciones por resultados.** Las 9 corridas (3 condiciones × 3 semillas) se reportan todas.
4. **No mezclar tensores E0 y seq para evaluar.** Usar eligible_indices para filtrar E0 a la misma cohorte que E1/E2.

### ✅ SÍ hacer

1. **Usar los mismos hiperparámetros y criterio de parada en E0, E1 y E2** (excepto los componentes específicos de cada estrategia de agregación).
2. **Usar las 3 semillas de entrenamiento: 42, 123, 2026** — aplicar las mismas a las 3 condiciones.
3. **Calcular ponderación de clases solo con SC-train:**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   weights = compute_class_weight('balanced', classes=np.arange(5), y=y_train)
   ```
4. **Evaluar en la cohorte común para comparaciones E0 vs E1 vs E2:**
   ```python
   # Filtrar E0 a las mismas épocas que E1/E2 pueden evaluar
   eligible = np.load("data/processed/seq/eligible_indices_test.npy")
   y_pred_e0_common = y_pred_e0[eligible]
   y_true_common = y_test[eligible]
   ```
5. **Guardar predicciones por sujeto** usando `groups_*.npy` — Luigi las necesita para Wilcoxon pareado.
6. **Registrar cada ejecución en MLflow** con: condición (E0/E1/E2), semilla, métricas, tiempo de entrenamiento/inferencia, parámetros del modelo.

---

## 6. Parámetros de normalización

Guardados en `data/processed/meta/norm_params.npz`:

```python
params = np.load("data/processed/meta/norm_params.npz")
media = params["media"]   # 3 valores — uno por canal
std = params["std"]       # 3 valores — uno por canal
```

Estos son los mismos valores que se usan para normalizar cualquier registro nuevo en la API de inferencia.

---

## 7. Inventario de registros

`data/processed/meta/inventory_records.csv` contiene una fila por cada uno de los 197 registros con:

| Campo | Descripción |
|-------|-------------|
| subject_id | Identificador del sujeto |
| subset | SC o ST |
| partition | train / val / test / st |
| psg_file, hyp_file | Archivos fuente EDF |
| ch_fpz_cz, ch_pz_oz, ch_eog | Canales verificados |
| fs_verified | Frecuencia de muestreo |
| epochs_raw | Épocas totales antes de filtros |
| epochs_excluded_mq | Excluidas por etiqueta M o ? |
| epochs_excluded_trim | Excluidas por trimming de Wake (30 min margen) |
| epochs_valid_e0 | Épocas válidas para E0 |
| epochs_excluded_border | Excluidas por borde sin contexto de 5 épocas |
| count_wake...count_rem | Distribución de clases |
| hash_psg, hash_hyp | MD5 de los archivos EDF fuente |

---

## 8. Validación

El script `test_data_tesis.py` ejecuta 21 verificaciones automatizadas:

**Fase 1 — E0 (12 checks):** shapes, dtypes, etiquetas, leakage SC interno, leakage SC-ST, normalización, NaN/Inf, consistencia X/y/groups, proporciones split, inventario vs tensores, norm_params, mini-batch PyTorch.

**Fase 2 — Secuencias (9 checks):** shapes seq, N'<N, etiquetas=época central, cohorte común, eligible_indices, exclusiones borde, sin cruce de límites, mini-batch seq, split_info completo.

Para ejecutar:
```bash
python -m src.data_tesis.test_data_tesis
```

**Resultado esperado: 21/21 checks pasados.**
