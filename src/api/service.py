"""
service.py
Capa de servicio que orquesta la carga de EDF, validación de señales y ejecución multimodelo.
"""

from collections import Counter
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from src.api.schemas import (
    ConsensusAnalysis,
    EDFInfoResponse,
    HypnogramEpoch,
    ModelPrediction,
    WindowAnalysisResponse,
    BatchAnalysisResponse,
)
from src.inference.predict import (
    CANALES_ML,
    DESCRIPCION_ETAPAS,
    ejecutar_inferencia_modelo,
    get_model_catalog,
    preparar_features,
)


def cargar_edf_desde_bytes(file_bytes: bytes, filename: str) -> mne.io.Raw:
    """
    Carga y parsea un archivo EDF desde un flujo de bytes en memoria.
    Valida preventivamente que no se trate de un archivo de hipnograma.
    """
    nombre_lower = filename.lower()
    if "hypnogram" in nombre_lower:
        raise ValueError(
            f"El archivo cargado '{filename}' es un hipnograma de anotaciones clínicas, no un registro PSG continuo. "
            "Para clasificar ventanas, debe cargar el archivo de señales correspondiente (ejemplo: 'SC4001E0-PSG.edf')."
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        return raw
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except PermissionError:
                pass


def validar_canales(raw) -> Tuple[List[str], List[str], bool]:
    """Retorna los canales presentes, los faltantes y si el archivo es válido."""
    presentes = [ch for ch in CANALES_ML if ch in raw.ch_names]
    faltantes = [ch for ch in CANALES_ML if ch not in raw.ch_names]
    is_valid = len(faltantes) == 0
    return presentes, faltantes, is_valid


def obtener_info_edf(raw, filename: str) -> EDFInfoResponse:
    """Extrae las dimensiones y metadatos del archivo EDF sin inferencia."""
    fs = float(raw.info["sfreq"])
    duracion = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
    total_ventanas = int(duracion // 30)

    presentes, faltantes, is_valid = validar_canales(raw)

    if is_valid:
        msg = f"Archivo válido con {total_ventanas} ventanas de 30s ({duracion / 3600:.2f} h) y todos los canales requeridos."
    else:
        msg = f"Archivo incompleto. Faltan los siguientes canales: {faltantes}."

    return EDFInfoResponse(
        filename=filename,
        sampling_rate_hz=round(fs, 2),
        duration_seconds=round(duracion, 2),
        duration_hours=round(duracion / 3600, 2),
        total_windows_30s=total_ventanas,
        channels_present=presentes,
        channels_missing=faltantes,
        is_valid=is_valid,
        message=msg,
    )


def resolver_modelos_solicitados(model_ids: Optional[List[str]]) -> List[str]:
    """
    Normaliza y valida los IDs de modelos solicitados en la selección múltiple.
    Soporta 'all' para activar todos los modelos disponibles en el catálogo.
    """
    catalog = get_model_catalog()
    modelos_disponibles = [k for k, v in catalog.items() if v["available"]]

    if not model_ids or len(model_ids) == 0:
        return modelos_disponibles

    # Si viene como lista con strings separados por comas, expandir
    modelos_limpios = []
    for item in model_ids:
        for m in item.replace(",", " ").split():
            clean = m.strip().lower()
            if clean:
                modelos_limpios.append(clean)

    if "all" in modelos_limpios or "*" in modelos_limpios:
        return modelos_disponibles

    modelos_finales = []
    modelos_no_disponibles = []

    for m in modelos_limpios:
        if m not in catalog:
            raise ValueError(
                f"Modelo '{m}' no reconocido. Opciones registradas: {list(catalog.keys())}"
            )
        if not catalog[m]["available"]:
            modelos_no_disponibles.append(m)
        else:
            if m not in modelos_finales:
                modelos_finales.append(m)

    if not modelos_finales:
        raise ValueError(
            f"Ninguno de los modelos solicitados ({modelos_limpios}) está disponible actualmente. "
            f"Modelos no disponibles: {modelos_no_disponibles}. "
            f"Disponibles: {modelos_disponibles}"
        )

    return modelos_finales


def analizar_ventana_edf(
    raw,
    filename: str,
    window_index: int,
    model_ids: Optional[List[str]] = None,
) -> WindowAnalysisResponse:
    """
    Ejecuta el análisis multimodelo sobre una ventana específica de 30 segundos.
    Calcula predicciones por modelo y consenso.
    """
    t_start = perf_counter()

    presentes, faltantes, is_valid = validar_canales(raw)
    if not is_valid:
        raise ValueError(
            f"No es posible clasificar la ventana. El EDF carece de los canales: {faltantes}."
        )

    fs = float(raw.info["sfreq"])
    duracion = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
    total_ventanas = int(duracion // 30)

    if total_ventanas == 0:
        raise ValueError("El archivo EDF dura menos de 30 segundos; no hay ventanas completas.")

    if window_index < 0 or window_index >= total_ventanas:
        raise ValueError(
            f"Índice de ventana {window_index} fuera de rango. El archivo tiene {total_ventanas} ventanas (0 a {total_ventanas - 1})."
        )

    inicio_segundo = float(window_index * 30.0)
    fin_segundo = inicio_segundo + 30.0

    modelos_a_ejecutar = resolver_modelos_solicitados(model_ids)

    # Optimización: si hay modelos tabulares (SVM, LightGBM), extraer features una sola vez
    features_df = None
    if any(m in ("svm", "lightgbm") for m in modelos_a_ejecutar):
        features_df = preparar_features(raw, inicio=inicio_segundo, duracion=30.0)

    catalog = get_model_catalog()
    predicciones: Dict[str, ModelPrediction] = {}
    etapas_lista: List[str] = []
    confianzas_lista: List[float] = []

    for m_id in modelos_a_ejecutar:
        res = ejecutar_inferencia_modelo(
            model_key=m_id,
            raw=raw,
            inicio=inicio_segundo,
            duracion=30.0,
            features_df=features_df,
        )

        prediccion_obj = ModelPrediction(
            model_id=res["model_id"],
            model_name=catalog[m_id]["name"],
            stage=res["stage"],
            stage_description=res["stage_description"],
            confidence=res["confidence"],
            probabilities=res["probabilities"],
        )

        predicciones[m_id] = prediccion_obj
        etapas_lista.append(res["stage"])
        confianzas_lista.append(res["confidence"])

    # ── Análisis de Consenso ──────────────────────────────────────────────────
    conteo = Counter(etapas_lista)
    etapa_consenso, mayor_frecuencia = conteo.most_common(1)[0]
    total_modelos = len(etapas_lista)

    porcentaje_acuerdo = round((mayor_frecuencia / total_modelos) * 100.0, 1)
    acuerdo_total = (mayor_frecuencia == total_modelos)
    media_confianza = round(float(np.mean(confianzas_lista)), 4) if confianzas_lista else 0.0

    consenso_obj = ConsensusAnalysis(
        consensus_stage=etapa_consenso,
        consensus_description=DESCRIPCION_ETAPAS.get(etapa_consenso, ""),
        agreement=acuerdo_total,
        agreement_percentage=porcentaje_acuerdo,
        mean_confidence=media_confianza,
    )

    t_total = (perf_counter() - t_start) * 1000.0

    return WindowAnalysisResponse(
        filename=filename,
        window_index=window_index,
        start_second=inicio_segundo,
        end_second=fin_segundo,
        models_analyzed=modelos_a_ejecutar,
        predictions=predicciones,
        consensus=consenso_obj,
        execution_time_ms=round(t_total, 2),
    )


def analizar_lote_edf(
    raw,
    filename: str,
    start_window: int = 0,
    max_windows: Optional[int] = 50,
    model_ids: Optional[List[str]] = None,
) -> BatchAnalysisResponse:
    """
    Analiza un rango de ventanas consecutivas para construir un hipnograma multimodelo.
    """
    t_start = perf_counter()

    presentes, faltantes, is_valid = validar_canales(raw)
    if not is_valid:
        raise ValueError(f"Faltan canales requeridos en el EDF: {faltantes}.")

    duracion = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
    total_disponibles = int(duracion // 30)

    if total_disponibles == 0:
        raise ValueError("El archivo EDF dura menos de 30 segundos.")

    inicio = max(0, start_window)
    limite = total_disponibles if max_windows is None else min(total_disponibles, inicio + max_windows)

    modelos_a_ejecutar = resolver_modelos_solicitados(model_ids)

    epocas: List[HypnogramEpoch] = []
    distribucion_por_modelo: Dict[str, Dict[str, int]] = {
        m: {c: 0 for c in ["Wake", "N1", "N2", "N3", "REM"]}
        for m in modelos_a_ejecutar
    }
    distribucion_consenso: Dict[str, int] = {c: 0 for c in ["Wake", "N1", "N2", "N3", "REM"]}

    for w_idx in range(inicio, limite):
        sec = float(w_idx * 30.0)
        features_df = None
        if any(m in ("svm", "lightgbm") for m in modelos_a_ejecutar):
            features_df = preparar_features(raw, inicio=sec, duracion=30.0)

        preds_w = {}
        for m in modelos_a_ejecutar:
            res = ejecutar_inferencia_modelo(m, raw, sec, 30.0, features_df)
            stage = res["stage"]
            preds_w[m] = stage
            distribucion_por_modelo[m][stage] += 1

        conteo = Counter(preds_w.values())
        consenso_stage = conteo.most_common(1)[0][0]
        distribucion_consenso[consenso_stage] += 1

        epocas.append(
            HypnogramEpoch(
                window_index=w_idx,
                start_second=sec,
                predictions=preds_w,
                consensus_stage=consenso_stage,
            )
        )

    t_total = (perf_counter() - t_start) * 1000.0

    return BatchAnalysisResponse(
        filename=filename,
        total_windows_analyzed=len(epocas),
        models_analyzed=modelos_a_ejecutar,
        epochs=epocas,
        stage_distribution_by_model=distribucion_por_modelo,
        consensus_distribution=distribucion_consenso,
        execution_time_ms=round(t_total, 2),
    )
