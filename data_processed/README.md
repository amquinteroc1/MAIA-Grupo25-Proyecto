# data_processed/

Dataset procesado para entrenamiento de CNN 1D multicanal.
Generado por: Diego Andrés Burbano — feature/diego-cnn-loader
Fecha: Sep 12, 2026

## Artefactos

| Archivo | Shape | Descripción |
|---|---|---|
| X_train.npy | (364642, 3, 3000) | Épocas train — 3 canales x 3000 muestras (30s @ 100Hz) |
| X_test.npy | (93010, 3, 3000) | Épocas test |
| y_train.npy | (364642,) | Etiquetas train: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM |
| y_test.npy | (93010,) | Etiquetas test |
| groups_train.npy | (364642,) | ID de sujeto por época (train) |
| groups_test.npy | (93010,) | ID de sujeto por época (test) |
| split_info.json | — | Metadata del split: sujetos, seed, distribución de clases |

## Canales (orden en eje 1)

- Canal 0: EEG Fpz-Cz (100 Hz)
- Canal 1: EEG Pz-Oz (100 Hz)
- Canal 2: EOG horizontal (100 Hz)

## Split

- Sujetos train: 157 / Sujetos test: 40
- Método: GroupShuffleSplit 80/20 por sujeto — sin data leakage
- Seed: 42

## Distribución de clases

| Clase | Total |
|---|---|
| Wake | 289,856 |
| N2 | 88,983 |
| REM | 34,184 |
| N1 | 25,175 |
| N3 | 19,454 |
| **Total** | **457,652** |

## Reproducción

```bash
source ~/env/bin/activate
dvc pull
```