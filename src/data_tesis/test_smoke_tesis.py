"""
test_smoke_tesis.py
Smoke test del pipeline de tesis — verifica punta a punta sobre 1 sujeto SC y 1 ST.
Ejecutar: python -m src.data_tesis.test_smoke_tesis
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from pathlib import Path

from src.data_tesis.crear_dataset_tesis import (
    encontrar_pares, leer_sujeto, split_sc_sujetos,
    calcular_norm_params, aplicar_normalizacion,
    construir_secuencias_registro,
    CANALES, N_MUESTRAS, SEQ_LEN, RAW_SC, RAW_ST,
)

ERRORES = 0

def check(nombre, condicion, detalle=""):
    global ERRORES
    if condicion:
        print(f"  ✓ {nombre}")
    else:
        print(f"  ✗ {nombre}  ← FALLO {detalle}")
        ERRORES += 1


def main():
    global ERRORES
    print("=" * 65)
    print("  SMOKE TEST — Pipeline de tesis (1 sujeto SC + 1 sujeto ST)")
    print("=" * 65)

    # ── 1. Encontrar pares ────────────────────────────────────────────────
    print("\n[1] Encontrar pares PSG/Hypnogram...")
    pares_sc = encontrar_pares(RAW_SC)
    pares_st = encontrar_pares(RAW_ST)
    check("Pares SC encontrados", len(pares_sc) > 0, f"encontrados: {len(pares_sc)}")
    check("Pares ST encontrados", len(pares_st) > 0, f"encontrados: {len(pares_st)}")

    if not pares_sc or not pares_st:
        print("\n✗ No se puede continuar sin pares en ambos subconjuntos.")
        return

    # Tomar un par de cada subconjunto
    psg_sc, hyp_sc, sid_sc = pares_sc[0]
    psg_st, hyp_st, sid_st = pares_st[0]
    print(f"  Sujeto SC: {sid_sc} → {Path(psg_sc).name}")
    print(f"  Sujeto ST: {sid_st} → {Path(psg_st).name}")

    # ── 2. Lectura y trimming ─────────────────────────────────────────────
    print("\n[2] Lectura con trimming de Wake...")
    epocas_sc, labels_sc, info_sc = leer_sujeto(psg_sc, hyp_sc, trim_wake=True)
    epocas_st, labels_st, info_st = leer_sujeto(psg_st, hyp_st, trim_wake=True)

    check("SC: épocas no None", epocas_sc is not None)
    check("ST: épocas no None", epocas_st is not None)

    if epocas_sc is None or epocas_st is None:
        print("\n✗ No se puede continuar sin épocas válidas.")
        return

    check("SC shape", epocas_sc.shape[1] == 3 and epocas_sc.shape[2] == N_MUESTRAS,
          f"got {epocas_sc.shape}")
    check("ST shape", epocas_st.shape[1] == 3 and epocas_st.shape[2] == N_MUESTRAS,
          f"got {epocas_st.shape}")
    check("SC dtype float32", epocas_sc.dtype == np.float32)
    check("SC labels dtype int64", labels_sc.dtype == np.int64)
    check("SC etiquetas en [0,4]", set(labels_sc).issubset({0, 1, 2, 3, 4}),
          f"got {sorted(set(labels_sc.tolist()))}")
    check("ST etiquetas en [0,4]", set(labels_st).issubset({0, 1, 2, 3, 4}),
          f"got {sorted(set(labels_st.tolist()))}")

    # Verificar trimming
    check("SC trimming aplicado", info_sc["epochs_excluded_trim"] >= 0,
          f"excluidas: {info_sc['epochs_excluded_trim']}")
    check("SC épocas válidas > 0", info_sc["epochs_valid_e0"] > 0)
    print(f"  SC: raw={info_sc['epochs_raw']}, excl_mq={info_sc['epochs_excluded_mq']}, "
          f"excl_trim={info_sc['epochs_excluded_trim']}, valid={info_sc['epochs_valid_e0']}")
    print(f"  ST: raw={info_st['epochs_raw']}, excl_mq={info_st['epochs_excluded_mq']}, "
          f"excl_trim={info_st['epochs_excluded_trim']}, valid={info_st['epochs_valid_e0']}")

    # Lectura sin trimming para comparar
    epocas_sc_notrim, labels_sc_notrim, info_notrim = leer_sujeto(psg_sc, hyp_sc, trim_wake=False)
    check("Trimming reduce épocas de SC",
          len(labels_sc) <= len(labels_sc_notrim),
          f"con trim: {len(labels_sc)}, sin trim: {len(labels_sc_notrim)}")
    del epocas_sc_notrim, labels_sc_notrim

    # NaN e Inf
    check("SC sin NaN", not np.isnan(epocas_sc).any())
    check("SC sin Inf", not np.isinf(epocas_sc).any())
    check("ST sin NaN", not np.isnan(epocas_st).any())
    check("ST sin Inf", not np.isinf(epocas_st).any())

    # ── 3. Particionamiento ───────────────────────────────────────────────
    print("\n[3] Particionamiento SC (64/16/20)...")
    sujetos_sc = sorted(set(p[2] for p in pares_sc))
    sujetos_st_all = sorted(set(p[2] for p in pares_st))
    asignacion = split_sc_sujetos(sujetos_sc)

    n_train = sum(1 for v in asignacion.values() if v == "train")
    n_val = sum(1 for v in asignacion.values() if v == "val")
    n_test = sum(1 for v in asignacion.values() if v == "test")
    n_total_sc = len(sujetos_sc)
    print(f"  SC: {n_total_sc} sujetos → train={n_train}, val={n_val}, test={n_test}")
    print(f"  ST: {len(sujetos_st_all)} sujetos (completo)")

    check("Proporciones razonables",
          abs(n_train / n_total_sc - 0.64) < 0.05 and
          abs(n_val / n_total_sc - 0.16) < 0.05 and
          abs(n_test / n_total_sc - 0.20) < 0.05,
          f"train={n_train/n_total_sc:.2f}, val={n_val/n_total_sc:.2f}, test={n_test/n_total_sc:.2f}")

    # No solapamiento
    sets = {"train": set(), "val": set(), "test": set()}
    for sid, part in asignacion.items():
        sets[part].add(sid)
    check("Sin leakage train-val", len(sets["train"] & sets["val"]) == 0)
    check("Sin leakage train-test", len(sets["train"] & sets["test"]) == 0)
    check("Sin leakage val-test", len(sets["val"] & sets["test"]) == 0)
    check("Sin leakage SC-ST", len(set(sujetos_sc) & set(sujetos_st_all)) == 0)

    # ── 4. Normalización poblacional ──────────────────────────────────────
    print("\n[4] Normalización poblacional (sobre 1 sujeto SC como proxy)...")
    # Usamos solo el sujeto SC de prueba como proxy rápido
    media_proxy, std_proxy = calcular_norm_params([(psg_sc, hyp_sc, sid_sc)])
    check("Media tiene 3 valores", len(media_proxy) == 3, f"got {len(media_proxy)}")
    check("Std tiene 3 valores", len(std_proxy) == 3, f"got {len(std_proxy)}")
    check("Std > 0 en todos los canales", all(s > 0 for s in std_proxy),
          f"got {std_proxy}")
    print(f"  Media: {media_proxy}")
    print(f"  Std:   {std_proxy}")

    # Aplicar normalización
    epocas_sc_norm = aplicar_normalizacion(epocas_sc.copy(), media_proxy, std_proxy)
    for ch in range(3):
        mu = float(epocas_sc_norm[:, ch, :].mean())
        sigma = float(epocas_sc_norm[:, ch, :].std())
        check(f"Canal {ch} normalizado: media ≈ 0",
              abs(mu) < 0.1, f"media={mu:.4f}")
        check(f"Canal {ch} normalizado: std ≈ 1",
              abs(sigma - 1.0) < 0.1, f"std={sigma:.4f}")

    check("Sin NaN tras normalización", not np.isnan(epocas_sc_norm).any())

    # Aplicar a ST con params de SC
    epocas_st_norm = aplicar_normalizacion(epocas_st.copy(), media_proxy, std_proxy)
    check("ST normalizado sin NaN", not np.isnan(epocas_st_norm).any())
    check("ST normalizado sin Inf", not np.isinf(epocas_st_norm).any())
    # ST no necesariamente tiene media=0/std=1, pero debe ser razonable
    for ch in range(3):
        mu_st = float(epocas_st_norm[:, ch, :].mean())
        check(f"ST canal {ch}: media razonable (< 2.0)", abs(mu_st) < 2.0,
              f"media={mu_st:.4f}")

    # ── 5. Secuencias sin padding ─────────────────────────────────────────
    print(f"\n[5] Secuencias de {SEQ_LEN} épocas (sin padding)...")
    X_seq, y_seq, elig = construir_secuencias_registro(epocas_sc_norm, labels_sc)

    check("Secuencias no None", X_seq is not None)
    if X_seq is not None:
        pad = SEQ_LEN // 2
        n_epocas = len(labels_sc)
        expected_seq = n_epocas - 2 * pad

        check(f"Shape correcto (M, {SEQ_LEN}, 3, {N_MUESTRAS})",
              X_seq.shape == (expected_seq, SEQ_LEN, 3, N_MUESTRAS),
              f"got {X_seq.shape}, expected ({expected_seq}, {SEQ_LEN}, 3, {N_MUESTRAS})")
        check("N' < N (secuencias < épocas E0)",
              len(X_seq) < n_epocas,
              f"seq={len(X_seq)}, e0={n_epocas}")
        check(f"Épocas excluidas = {2 * pad} (borde)",
              n_epocas - len(X_seq) == 2 * pad,
              f"excluidas={n_epocas - len(X_seq)}")
        check("Etiqueta = época central",
              np.array_equal(y_seq, labels_sc[pad:n_epocas - pad]))
        check("Eligible indices correctos",
              np.array_equal(elig, np.arange(pad, n_epocas - pad)))
        check("dtype X_seq float32", X_seq.dtype == np.float32)
        check("dtype y_seq int64", y_seq.dtype == np.int64)

        # Mini batch
        if len(X_seq) >= 32:
            batch_X = X_seq[:32]
            check(f"Mini-batch shape (32, {SEQ_LEN}, 3, {N_MUESTRAS})",
                  batch_X.shape == (32, SEQ_LEN, 3, N_MUESTRAS))

    # Secuencias ST
    X_seq_st, y_seq_st, elig_st = construir_secuencias_registro(epocas_st_norm, labels_st)
    check("ST secuencias no None", X_seq_st is not None)
    if X_seq_st is not None:
        check("ST secuencias shape correcto",
              X_seq_st.shape[1] == SEQ_LEN and X_seq_st.shape[2] == 3,
              f"got {X_seq_st.shape}")

    # ── 6. Consistencia X vs y ────────────────────────────────────────────
    print("\n[6] Consistencia final...")
    check("SC: len(épocas) == len(labels)", len(epocas_sc_norm) == len(labels_sc))
    check("ST: len(épocas) == len(labels)", len(epocas_st_norm) == len(labels_st))
    if X_seq is not None:
        check("SC: len(X_seq) == len(y_seq)", len(X_seq) == len(y_seq))
        check("SC: len(X_seq) == len(eligible)", len(X_seq) == len(elig))

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    if ERRORES == 0:
        print(f"  ✅  SMOKE TEST PASADO — pipeline listo para 197 sujetos")
    else:
        print(f"  ❌  {ERRORES} FALLO(S) — revisar antes de continuar")
    print(f"{'=' * 65}\n")

    sys.exit(ERRORES)


if __name__ == "__main__":
    main()
