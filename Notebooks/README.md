# Notebooks/ — Cuadernos de Entrenamiento, Análisis y Validación

Esta carpeta contiene los cuadernos reproducibles correspondientes a las tres arquitecturas profundas evaluadas sobre **Sleep-EDF Expanded**:

| Cuaderno | Arquitectura | Descripción |
| :--- | :--- | :--- |
| `train_e0.ipynb` | **E0 (`SleepCNN1D`)** | Entrenamiento y evaluación del modelo baseline por época individual de 30 s (`3×3.000`). |
| `train_e1.ipynb` | **E1 (`SleepCNN1D_E1`)** | Entrenamiento con contexto temporal de 5 épocas (`150 s`) mediante concatenación de embeddings. |
| `train_e2.ipynb` | **E2 (`SleepCNN1D_E2`)** | Entrenamiento del modelo ganador con capa recurrente **GRU (256 u.)** sobre 5 épocas consecutivas. |

## Scripts Complementarios de Pruebas y Análisis

- **Validación de tensores y particiones por sujeto:** `python -m src.data_tesis.test_data_tesis`
- **Evaluación externa (Sleep-Telemetry) y pruebas estadísticas Wilcoxon:** `python -m src.evaluation.evaluacion_st` y `python -m src.evaluation.wilcoxon_test`
- **Matrices de confusión:** `python -m src.evaluation.generar_matrices_confusion`
- **Pruebas de integración API + UI:** `pytest tests/test_api.py -v`
