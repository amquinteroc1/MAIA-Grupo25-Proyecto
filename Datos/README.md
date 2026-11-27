# Datos/ — Estructura y Metadatos de Polisomnografía (Sleep-EDF Expanded)

Esta carpeta documenta la estructura de datos clínicos, distribución de registros y parámetros oficiales de normalización utilizados en el proyecto, provenientes de **[PhysioNet Sleep-EDF Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)**.

## Archivos Incluidos

| Archivo | Descripción |
| :--- | :--- |
| `norm_params.npz` | Media (`media`) y desviación estándar (`std`) por canal calculadas exclusivamente sobre `SC-train`, aplicadas como Z-score en entrenamiento, evaluación e inferencia (`E0/E1/E2`). |
| `sleep_edf_records.csv` | Inventario tabular de los **197 registros PSG** (`153` Sleep-Cassette + `44` Sleep-Telemetry), duración, frecuencia de muestreo y presencia de canales. |
| `sleep_edf_summary.json` | Resumen estructurado del análisis exploratorio (EDA) y validación de integridad de señales EDF. |

## Esquema de Señales y Clases (Estándar R&K / AASM)

- **Canales fisiológicos de entrada (`100 Hz`):** `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`.
- **Ventana clínica (época):** `30 s` (`3.000` muestras por canal).
- **Tensores de entrada:** `(N, 3, 3000)` para `E0` y `(N, 5, 3, 3000)` (secuencia de 5 épocas consecutivas `[t-2..t+2]`, `150 s`) para `E1` y `E2`.
- **Clases canónicas:** `0: Wake` (`W`), `1: N1` (`1`), `2: N2` (`2`), `3: N3` (`3` + `4`), `4: REM` (`R`).

## Particiones Estrictas por Sujeto

| Partición | Sujetos | Registros | Épocas `E0` | Épocas `E1/E2` | Uso |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `SC-train` | 49 | 96 | 120.175 | 119.364 | Entrenamiento y ajuste de `norm_params.npz` |
| `SC-val` | 13 | 25 | 35.149 | 34.981 | Selección de hiperparámetros y early stopping |
| `SC-test` | 16 | 32 | 40.145 | 39.961 | Evaluación interna sin fuga de sujetos |
| `ST` (externo) | 22 | 44 | 42.467 | 42.003 | Evaluación externa cross-domain |

Los tensores completos y metadatos (`split_info.json`, `group_mappings.json`) se encuentran en Hugging Face Hub ([`maia-grupo25/sleep-edfx-data`](https://huggingface.co/datasets/maia-grupo25/sleep-edfx-data)) y pueden reconstruirse con:

```powershell
python -m src.data_tesis.crear_dataset_tesis --fase todo
python -m src.data_tesis.test_data_tesis
```
