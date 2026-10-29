"""
test_data_tesis.py
Validación completa de los artefactos generados por crear_dataset_tesis.py.
21 checks que cubren el protocolo experimental unificado.

Ejecutar: python -m src.data_tesis.test_data_tesis
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E0_DIR = ROOT / "data" / "processed" / "e0"
SEQ_DIR = ROOT / "data" / "processed" / "seq"
META_DIR = ROOT / "data" / "processed" / "meta"

PARTICIONES = ["train", "val", "test", "st"]
ERRORES = 0
CHECK_NUM = 0


def check(nombre, condicion, detalle=""):
    global ERRORES, CHECK_NUM
    CHECK_NUM += 1
    if condicion:
        print(f"  [{CHECK_NUM:2d}] ✓ {nombre}")
    else:
        print(f"  [{CHECK_NUM:2d}] ✗ {nombre}  ← FALLO {detalle}")
        ERRORES += 1


def main():
    global ERRORES, CHECK_NUM

    print("=" * 70)
    print("  VALIDACIÓN COMPLETA — Artefactos del proyecto de grado")
    print("  21 checks contra protocolo experimental unificado")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # FASE 1 — TENSORES E0 (checks 1–12)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── FASE 1: Tensores E0 ──────────────────────────────────────")

    # Cargar todos los tensores E0
    e0 = {}
    for p in PARTICIONES:
        e0[p] = {
            "X": np.load(E0_DIR / f"X_{p}.npy", mmap_mode="r"),
            "y": np.load(E0_DIR / f"y_{p}.npy"),
            "g": np.load(E0_DIR / f"groups_{p}.npy"),
        }

    # [1] Shapes
    print()
    all_shapes_ok = True
    for p in PARTICIONES:
        X, y, g = e0[p]["X"], e0[p]["y"], e0[p]["g"]
        if X.shape[1] != 3 or X.shape[2] != 3000:
            all_shapes_ok = False
        if len(y) != X.shape[0] or len(g) != X.shape[0]:
            all_shapes_ok = False
    shapes_detail = {p: e0[p]["X"].shape for p in PARTICIONES}
    check("Shapes E0: (N, 3, 3000) en 4 conjuntos", all_shapes_ok, str(shapes_detail))

    # [2] Dtypes
    dtypes_ok = all(
        e0[p]["X"].dtype == np.float32 and
        e0[p]["y"].dtype == np.int64 and
        e0[p]["g"].dtype == np.int64
        for p in PARTICIONES
    )
    check("Dtypes: X=float32, y=int64, groups=int64", dtypes_ok)

    # [3] Etiquetas en [0,4]
    labels_ok = all(
        set(np.unique(e0[p]["y"])).issubset({0, 1, 2, 3, 4})
        for p in PARTICIONES
    )
    check("Etiquetas en [0,1,2,3,4] en 4 conjuntos", labels_ok)

    # [4] Data leakage SC interno
    suj_train = set(np.unique(e0["train"]["g"]))
    suj_val = set(np.unique(e0["val"]["g"]))
    suj_test = set(np.unique(e0["test"]["g"]))
    no_leak_tv = len(suj_train & suj_val) == 0
    no_leak_tt = len(suj_train & suj_test) == 0
    no_leak_vt = len(suj_val & suj_test) == 0
    check("Sin leakage SC: train∩val=∅, train∩test=∅, val∩test=∅",
          no_leak_tv and no_leak_tt and no_leak_vt,
          f"t∩v={len(suj_train & suj_val)}, t∩te={len(suj_train & suj_test)}, v∩te={len(suj_val & suj_test)}")

    # [5] Data leakage SC vs ST
    suj_st = set(np.unique(e0["st"]["g"]))
    suj_sc = suj_train | suj_val | suj_test
    no_leak_sc_st = len(suj_sc & suj_st) == 0
    check("Sin leakage SC vs ST: SC∩ST=∅", no_leak_sc_st,
          f"compartidos: {len(suj_sc & suj_st)}")

    # [6] Normalización SC-train ≈ media 0, std 1
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(e0["train"]["X"].shape[0], min(1000, e0["train"]["X"].shape[0]), replace=False)
    sample_train = np.array(e0["train"]["X"][idx_sample])
    norm_ok = True
    norm_details = []
    for ch in range(3):
        mu = float(sample_train[:, ch, :].mean())
        std = float(sample_train[:, ch, :].std())
        norm_details.append(f"ch{ch}: μ={mu:.4f} σ={std:.4f}")
        if abs(mu) > 0.1 or abs(std - 1.0) > 0.15:
            norm_ok = False
    check("Normalización SC-train: media≈0, std≈1 por canal", norm_ok,
          " | ".join(norm_details))

    # [7] NaN e Inf
    nan_inf_ok = True
    for p in PARTICIONES:
        idx_s = rng.choice(e0[p]["X"].shape[0], min(500, e0[p]["X"].shape[0]), replace=False)
        sample = np.array(e0[p]["X"][idx_s])
        if np.isnan(sample).any() or np.isinf(sample).any():
            nan_inf_ok = False
    check("Sin NaN ni Inf (muestra 500 épocas × 4 conjuntos)", nan_inf_ok)

    # [8] Consistencia X vs y vs groups
    consist_ok = all(
        e0[p]["X"].shape[0] == len(e0[p]["y"]) == len(e0[p]["g"])
        for p in PARTICIONES
    )
    check("Consistencia: len(X)==len(y)==len(groups) en 4 conjuntos", consist_ok)

    # [9] Proporciones del split
    n_suj_train = len(suj_train)
    n_suj_val = len(suj_val)
    n_suj_test = len(suj_test)
    n_sc_total = n_suj_train + n_suj_val + n_suj_test
    prop_ok = (
        abs(n_suj_train / n_sc_total - 0.64) < 0.05 and
        abs(n_suj_val / n_sc_total - 0.16) < 0.05 and
        abs(n_suj_test / n_sc_total - 0.20) < 0.05
    )
    check(f"Proporciones: train={n_suj_train}/{n_sc_total}={n_suj_train/n_sc_total:.2f}, "
          f"val={n_suj_val}/{n_sc_total}={n_suj_val/n_sc_total:.2f}, "
          f"test={n_suj_test}/{n_sc_total}={n_suj_test/n_sc_total:.2f}",
          prop_ok)

    # [10] Inventario vs tensores
    inv_path = META_DIR / "inventory_records.csv"
    inv_exists = inv_path.exists()
    inv_match = False
    if inv_exists:
        with open(inv_path, "r") as f:
            reader = csv.DictReader(f)
            inv_rows = list(reader)
        inv_totals = {"train": 0, "val": 0, "test": 0, "st": 0}
        for row in inv_rows:
            part = row["partition"]
            inv_totals[part] += int(row["epochs_valid_e0"])
        inv_match = all(
            inv_totals[p] == e0[p]["X"].shape[0]
            for p in PARTICIONES
        )
    check("Inventario vs tensores: épocas coinciden",
          inv_exists and inv_match,
          f"inv={inv_totals if inv_exists else 'NO EXISTE'} vs tensores={{{p}: e0[p]['X'].shape[0] for p in PARTICIONES}})")

    # [11] norm_params.npz existe y es válido
    np_path = META_DIR / "norm_params.npz"
    np_exists = np_path.exists()
    np_valid = False
    if np_exists:
        np_data = np.load(np_path)
        media = np_data["media"]
        std = np_data["std"]
        np_valid = (
            len(media) == 3 and len(std) == 3 and
            not np.isnan(media).any() and not np.isnan(std).any() and
            all(s > 0 for s in std)
        )
    check("norm_params.npz: 3 medias, 3 stds, sin NaN, std>0",
          np_exists and np_valid,
          f"existe={np_exists}")

    # [12] Mini-batch PyTorch
    batch_X = np.array(e0["train"]["X"][:32])
    batch_ok = batch_X.shape == (32, 3, 3000) and batch_X.dtype == np.float32
    check("Mini-batch E0: shape (32,3,3000) float32 → PyTorch compatible", batch_ok,
          f"shape={batch_X.shape} dtype={batch_X.dtype}")

    # ══════════════════════════════════════════════════════════════════════
    # FASE 2 — SECUENCIAS E1/E2 (checks 13–21)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── FASE 2: Secuencias E1/E2 ─────────────────────────────────")

    seq = {}
    elig = {}
    for p in PARTICIONES:
        seq[p] = {
            "X": np.load(SEQ_DIR / f"X_seq_{p}.npy", mmap_mode="r"),
            "y": np.load(SEQ_DIR / f"y_seq_{p}.npy"),
            "g": np.load(SEQ_DIR / f"groups_seq_{p}.npy"),
        }
        elig[p] = np.load(SEQ_DIR / f"eligible_indices_{p}.npy")

    # [13] Shapes secuencias
    seq_shapes_ok = all(
        seq[p]["X"].shape[1] == 5 and
        seq[p]["X"].shape[2] == 3 and
        seq[p]["X"].shape[3] == 3000
        for p in PARTICIONES
    )
    seq_shapes_detail = {p: seq[p]["X"].shape for p in PARTICIONES}
    check("Shapes seq: (N', 5, 3, 3000) en 4 conjuntos", seq_shapes_ok,
          str(seq_shapes_detail))

    # [14] N' < N
    n_prime_lt_n = all(
        seq[p]["X"].shape[0] < e0[p]["X"].shape[0]
        for p in PARTICIONES
    )
    check("N' < N: secuencias < épocas E0 en cada partición", n_prime_lt_n,
          {p: f"{seq[p]['X'].shape[0]} < {e0[p]['X'].shape[0]}" for p in PARTICIONES})

    # [15] Etiquetas secuencias = época central
    labels_central_ok = True
    for p in PARTICIONES:
        y_e0_eligible = e0[p]["y"][elig[p]]
        if not np.array_equal(seq[p]["y"], y_e0_eligible):
            labels_central_ok = False
    check("Etiquetas seq corresponden a época central (via eligible_indices)", labels_central_ok)

    # [16] Cohorte común: len(X_seq) == len(eligible_indices)
    cohort_ok = all(
        seq[p]["X"].shape[0] == len(elig[p])
        for p in PARTICIONES
    )
    check("Cohorte común: len(X_seq)==len(eligible_indices) por partición", cohort_ok,
          {p: f"seq={seq[p]['X'].shape[0]} elig={len(elig[p])}" for p in PARTICIONES})

    # [17] Eligible indices: rango válido y sin duplicados
    elig_valid = True
    for p in PARTICIONES:
        e = elig[p]
        n_e0 = e0[p]["X"].shape[0]
        if e.min() < 0 or e.max() >= n_e0:
            elig_valid = False
        if len(np.unique(e)) != len(e):
            elig_valid = False
    check("Eligible indices: rango [0, N_e0) y sin duplicados", elig_valid)

    # [18] Exclusiones por borde vs inventario
    if inv_exists:
        inv_border_by_part = {"train": 0, "val": 0, "test": 0, "st": 0}
        for row in inv_rows:
            part = row["partition"]
            inv_border_by_part[part] += int(row.get("epochs_excluded_border", 0))

        actual_border = {
            p: e0[p]["X"].shape[0] - seq[p]["X"].shape[0]
            for p in PARTICIONES
        }
        # El inventario puede no tener epochs_excluded_border por sujeto
        # (se calcula a nivel de partición en split_info), así que verificamos
        # contra split_info si el inventario no lo tiene
        si_path = META_DIR / "split_info.json"
        border_check_ok = True
        if si_path.exists():
            with open(si_path) as f:
                si = json.load(f)
            if "exclusiones_borde" in si:
                for p in PARTICIONES:
                    expected = si["exclusiones_borde"].get(p, {}).get("excluded_border", -1)
                    if expected != actual_border[p]:
                        border_check_ok = False
        check("Exclusiones borde: split_info coincide con diferencia E0-seq",
              border_check_ok,
              f"actual={actual_border}")
    else:
        check("Exclusiones borde: inventario no existe", False)

    # [19] Sin cruce de límites de registro dentro de secuencias
    # Verificar que groups es constante dentro de cada secuencia (muestreo)
    no_cross_ok = True
    for p in ["train", "test"]:  # muestra en train y test
        n_seq = seq[p]["X"].shape[0]
        sample_idx = rng.choice(n_seq, min(200, n_seq), replace=False)
        for idx in sample_idx:
            # Los groups de las 5 épocas deberían ser iguales
            # Verificamos indirectamente: eligible_indices consecutivos deben
            # pertenecer al mismo grupo en E0
            center = elig[p][idx]
            indices_ventana = np.arange(center - 2, center + 3)
            if indices_ventana[0] < 0 or indices_ventana[-1] >= len(e0[p]["g"]):
                no_cross_ok = False
                break
            groups_ventana = e0[p]["g"][indices_ventana]
            if len(np.unique(groups_ventana)) != 1:
                no_cross_ok = False
                break
    check("Sin cruce de límites: groups constante en ventanas (muestra 200)", no_cross_ok)

    # [20] Mini-batch secuencias PyTorch
    batch_seq = np.array(seq["train"]["X"][:32])
    batch_seq_ok = batch_seq.shape == (32, 5, 3, 3000) and batch_seq.dtype == np.float32
    check("Mini-batch seq: shape (32,5,3,3000) float32 → PyTorch compatible",
          batch_seq_ok, f"shape={batch_seq.shape} dtype={batch_seq.dtype}")

    # [21] split_info.json completo
    si_path = META_DIR / "split_info.json"
    si_complete = False
    if si_path.exists():
        with open(si_path) as f:
            si = json.load(f)
        required_keys = ["split_seed", "split_proporciones", "canales", "sujetos",
                         "n_sujetos", "conteos_e0", "conteos_seq",
                         "exclusiones_borde", "norm_params_file"]
        si_complete = all(k in si for k in required_keys)
        missing = [k for k in required_keys if k not in si]
    check("split_info.json: todas las claves requeridas presentes",
          si_complete,
          f"faltan: {missing}" if not si_complete and si_path.exists() else "")

    # ══════════════════════════════════════════════════════════════════════
    # RESUMEN
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    if ERRORES == 0:
        print(f"  ✅  {CHECK_NUM} / {CHECK_NUM} CHECKS PASADOS")
        print(f"  Pipeline validado contra protocolo experimental unificado")
    else:
        print(f"  ❌  {ERRORES} FALLO(S) de {CHECK_NUM} checks")
        print(f"  Revisar antes de entregar a Omar")
    print(f"{'=' * 70}\n")

    # Imprimir resumen de datos
    print("── Resumen de artefactos ────────────────────────────────────")
    for p in PARTICIONES:
        n_e0 = e0[p]["X"].shape[0]
        n_seq = seq[p]["X"].shape[0]
        n_suj = len(np.unique(e0[p]["g"]))
        print(f"  {p:6s}: {n_e0:>7,} épocas E0 | {n_seq:>7,} secuencias | {n_suj:>3} sujetos | borde excluido: {n_e0 - n_seq}")
    print()

    sys.exit(ERRORES)


if __name__ == "__main__":
    main()
