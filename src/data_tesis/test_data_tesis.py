"""Validacion integral de datos de tesis E0 / E1 / E2.

Reemplaza src/data_tesis/test_data_tesis.py del microproyecto.
Ejecucion desde la raiz del proyecto:
    python -m src.data_tesis.test_data_tesis

No modifica el dataset ni carga los arrays de senales completos en RAM.
Requiere los artefactos generados con `--fase e0` y `--fase seq`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.data_tesis.crear_dataset_tesis import (
    CLASE_NOMBRES,
    N_MUESTRAS,
    PROTOCOL_VERSION,
    SEQ_LEN,
    SPLIT_SEED,
    _centros_validos,
    split_sc_sujetos,
)

ROOT = Path(__file__).resolve().parents[2]
E0 = ROOT / "data" / "processed" / "e0"
SEQ = ROOT / "data" / "processed" / "seq"
META = ROOT / "data" / "processed" / "meta"
PARTS = ("train", "val", "test", "st")
EXPECTED_RECORDS = {"train": 96, "val": 25, "test": 32, "st": 44}
EXPECTED_SUBJECTS = {"train": 49, "val": 13, "test": 16, "st": 22}
CLASS_NAMES = [CLASE_NOMBRES[i] for i in range(5)]


class Checker:
    def __init__(self):
        self.passed = 0

    def check(self, description: str, condition: bool, detail: str = ""):
        if not bool(condition):
            raise AssertionError(f"FALLO [{description}]" + (f": {detail}" if detail else ""))
        self.passed += 1
        print(f"  [{self.passed:02d}] OK: {description}", flush=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npy(path: Path, mmap: bool = False):
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Genera E0 y secuencias primero.")
    return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)


def sample_indices(n: int, max_n: int = 48):
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, min(max_n, n), dtype=np.int64))


def validate():
    c = Checker()
    print("Validacion E0 / E1 / E2: protocolo sujeto-exclusivo y cronologico", flush=True)
    print(f"Raiz: {ROOT}\n", flush=True)

    required_meta = [META / "inventory_records.csv", META / "split_info.json",
                     META / "group_mappings.json", META / "norm_params.npz"]
    required_e0 = [E0 / f"{prefix}_{p}.npy" for p in PARTS
                   for prefix in ("X", "y", "groups", "record_ids", "epoch_starts")]
    required_seq = [SEQ / f"{prefix}_{p}.npy" for p in PARTS
                    for prefix in ("X_seq", "y_seq", "groups_seq", "eligible_indices")]
    missing = [str(p) for p in required_meta + required_e0 + required_seq if not p.is_file()]
    c.check("Todos los artefactos esperados existen", not missing,
            "Archivos ausentes:\n" + "\n".join(missing))

    info = load_json(META / "split_info.json")
    mappings = load_json(META / "group_mappings.json")
    with (META / "inventory_records.csv").open(newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    subject_map = mappings["subject_to_group"]
    record_map = mappings["record_to_group"]
    c.check("Protocolo v2 en metadatos", info.get("protocol_version") == PROTOCOL_VERSION
            and mappings.get("protocol_version") == PROTOCOL_VERSION)
    c.check("Semilla de particion fija", info.get("split_seed") == SPLIT_SEED)
    c.check("Inventario completo: 197 registros", len(inventory) == 197)
    c.check("Identificadores de registros unicos", len({r["record_id"] for r in inventory}) == 197)
    c.check("Cien personas reales", len({r["subject_id"] for r in inventory}) == 100)
    c.check("Grupos numericos reproducibles", subject_map ==
            {sid: i + 1 for i, sid in enumerate(sorted(subject_map))}
            and record_map == {rid: i + 1 for i, rid in enumerate(sorted(record_map))})
    c.check("Cada registro corresponde a su persona", all(
            r["record_id"][:5] == r["subject_id"] for r in inventory))

    subjects = {p: {r["subject_id"] for r in inventory if r["partition"] == p}
                for p in PARTS}
    records = {p: [r for r in inventory if r["partition"] == p] for p in PARTS}
    c.check("Personas por conjunto: 49/13/16/22", all(
            len(subjects[p]) == EXPECTED_SUBJECTS[p] for p in PARTS))
    c.check("Registros por conjunto: 96/25/32/44", all(
            len(records[p]) == EXPECTED_RECORDS[p] for p in PARTS))
    c.check("Sujetos SC y ST correctamente diferenciados", all(
            (sid.startswith("ST") if p == "st" else sid.startswith("SC"))
            for p in PARTS for sid in subjects[p]))
    c.check("Ausencia total de leakage entre personas", all(
            not (subjects[p] & subjects[q]) for i, p in enumerate(PARTS)
            for q in PARTS[i + 1:]))
    c.check("Asignacion reproducible desde los sujetos SC", all(
            split_sc_sujetos(subjects["train"] | subjects["val"] | subjects["test"])[sid] == p
            for p in ("train", "val", "test") for sid in subjects[p]))
    c.check("Inventario y split_info coinciden", all(
            set(info["sujetos"][p]) == subjects[p]
            and set(info["registros"][p]) == {r["record_id"] for r in records[p]}
            and info["n_sujetos"][p] == len(subjects[p])
            and info["n_registros"][p] == len(records[p])
            for p in PARTS))
    c.check("Mapeos de grupos coinciden con el inventario", all(
            int(r["subject_group_id"]) == subject_map[r["subject_id"]]
            and int(r["record_group_id"]) == record_map[r["record_id"]]
            for r in inventory))

    with np.load(META / "norm_params.npz", allow_pickle=False) as norm:
        mean = norm["media"]
        std = norm["std"]
    c.check("Parametros de normalizacion validos", mean.shape == (3,) and
            std.shape == (3,) and np.isfinite(mean).all() and
            np.isfinite(std).all() and (std > 0).all())
    c.check("Normalizacion coherente con split_info", np.allclose(
            mean, info["norm_media_por_canal"], rtol=1e-8, atol=1e-12)
            and np.allclose(std, info["norm_std_por_canal"], rtol=1e-8, atol=1e-12))
    c.check("Longitud de secuencia y frecuencia", info.get("seq_len") == SEQ_LEN
            and info.get("n_muestras_epoca") == N_MUESTRAS
            and info.get("sample_rate_hz") == 100)

    # La prueba sintética detecta que no se permiten ventanas a traves de huecos.
    synthetic_starts = np.arange(10, dtype=np.int64) * N_MUESTRAS
    synthetic_starts[5:] += N_MUESTRAS
    c.check("Control sintetico de discontinuidades temporales",
            _centros_validos(synthetic_starts).tolist() == [2, 7])

    all_gids = {}
    for part in PARTS:
        print(f"\n--- {part.upper()} ---", flush=True)
        X = load_npy(E0 / f"X_{part}.npy", mmap=True)
        y = load_npy(E0 / f"y_{part}.npy", mmap=True)
        groups = load_npy(E0 / f"groups_{part}.npy", mmap=True)
        recids = load_npy(E0 / f"record_ids_{part}.npy", mmap=True)
        starts = load_npy(E0 / f"epoch_starts_{part}.npy", mmap=True)
        n = len(y)
        c.check(f"{part} E0: dimensiones y dtypes", X.shape == (n, 3, N_MUESTRAS)
                and X.dtype == np.float32 and y.dtype == np.int64
                and groups.dtype == np.int64 and recids.dtype == np.int64
                and starts.dtype == np.int64 and len(groups) == len(recids) == len(starts) == n)
        c.check(f"{part} E0: etiquetas en 0..4", n > 0 and
                np.isin(np.unique(y), np.arange(5)).all())
        c.check(f"{part} E0: IDs de persona reales", set(np.unique(groups).tolist()) ==
                {subject_map[sid] for sid in subjects[part]})
        all_gids[part] = set(np.unique(groups).tolist())
        c.check(f"{part} E0: IDs de registro reales", set(np.unique(recids).tolist()) ==
                {record_map[r["record_id"]] for r in records[part]})
        c.check(f"{part} E0: conteos e inventario", n ==
                sum(int(r["epochs_valid_e0"]) for r in records[part])
                and n == info["conteos_e0"][part]["total"])
        counts = np.bincount(y, minlength=5)
        c.check(f"{part} E0: distribucion por clase", all(
                int(info["conteos_e0"][part]["por_clase"][CLASS_NAMES[k]]) == int(counts[k])
                for k in range(5)))
        c.check(f"{part} E0: senales finitas en muestra", all(
                np.isfinite(X[int(i)]).all() for i in sample_indices(n, 32)))

        X_seq = load_npy(SEQ / f"X_seq_{part}.npy", mmap=True)
        y_seq = load_npy(SEQ / f"y_seq_{part}.npy", mmap=True)
        g_seq = load_npy(SEQ / f"groups_seq_{part}.npy", mmap=True)
        eligible = load_npy(SEQ / f"eligible_indices_{part}.npy", mmap=True)
        m = len(eligible)
        c.check(f"{part} secuencias: dimensiones y dtypes",
                X_seq.shape == (m, 5, 3, N_MUESTRAS) and X_seq.dtype == np.float32
                and y_seq.dtype == np.int64 and g_seq.dtype == np.int64
                and eligible.dtype == np.int64 and len(y_seq) == len(g_seq) == m)
        c.check(f"{part} secuencias: indices validos y sin duplicados", m > 0
                and eligible.min() >= 2 and eligible.max() < n - 2
                and len(np.unique(eligible)) == m)
        c.check(f"{part} secuencias: etiquetas centrales exactas",
                np.array_equal(y_seq, y[eligible]))
        c.check(f"{part} secuencias: personas centrales exactas",
                np.array_equal(g_seq, groups[eligible]))
        offsets = np.arange(-2, 3, dtype=np.int64)
        neighbors = eligible[:, None] + offsets
        c.check(f"{part} ventanas: sin cruces de registro", np.all(
                recids[neighbors] == recids[eligible, None]))
        c.check(f"{part} ventanas: tiempo continuo 30 s", np.all(
                np.diff(starts[neighbors], axis=1) == N_MUESTRAS))
        c.check(f"{part} ventanas: grupos estables", np.all(
                groups[neighbors] == groups[eligible, None]))

        # Verificacion exhaustiva de centros y conteos por registro.
        expected_centers = []
        detail_border = 0
        detail_gaps = 0
        for row in records[part]:
            rid = row["record_id"]
            r_gid = record_map[rid]
            idx = np.flatnonzero(recids == r_gid)
            s = starts[idx]
            center_local = _centros_validos(s)
            expected_centers.extend(idx[center_local].tolist())
            excluded_border = min(len(idx), SEQ_LEN - 1)
            excluded_gap = max(0, len(idx) - excluded_border - len(center_local))
            detail_border += excluded_border
            detail_gaps += excluded_gap
            if (len(idx) != int(row["epochs_valid_e0"])
                    or len(center_local) != int(row["epochs_valid_seq"])
                    or excluded_border != int(row["epochs_excluded_border"])
                    or excluded_gap != int(row["epochs_excluded_discontinuity"])):
                raise AssertionError(f"Inventario E0/SEQ/EXCLUSIONES incorrecto: {rid}")
            if len(idx) and (not np.all(np.diff(idx) == 1)
                             or not np.all(groups[idx] == subject_map[row["subject_id"]])
                             or not np.all(np.diff(s) >= N_MUESTRAS)):
                raise AssertionError(f"Registro fragmentado/mezclado/solapado: {rid}")
            perclass = np.bincount(y[idx], minlength=5)
            for k in range(5):
                if int(row[f"count_{CLASS_NAMES[k].lower()}"]) != int(perclass[k]):
                    raise AssertionError(f"Conteo de clase incorrecto en: {rid}")
        c.check(f"{part} secuencias: centros exhaustivos por registro",
                np.array_equal(eligible, np.asarray(expected_centers, dtype=np.int64)))
        c.check(f"{part} inventario: exclusiones borde y discontinuidad",
                n - m == detail_border + detail_gaps and
                info["exclusiones_detalle"][part] == {
                    "border": detail_border, "discontinuity": detail_gaps})
        c.check(f"{part} secuencias: total y clases en metadatos",
                m == info["conteos_seq"][part]["total"] and
                all(int(info["conteos_seq"][part]["por_clase"][CLASS_NAMES[k]]) ==
                    int(np.count_nonzero(y_seq == k)) for k in range(5)))
        for i in sample_indices(m, 24):
            j = int(i)
            if not np.array_equal(X_seq[j], X[neighbors[j]]):
                raise AssertionError(f"Señal no coincide en {part}, secuencia {j}")
        c.check(f"{part} secuencias: señales completas (muestra de 24)", True)
        print(f"  Resultado: {n:,} E0, {m:,} secuencias, "
              f"{detail_border} bordes, {detail_gaps} discontinuidades")

    c.check("Sin identificadores numericos compartidos entre conjuntos", all(
            not all_gids[p].intersection(all_gids[q])
            for i, p in enumerate(PARTS) for q in PARTS[i + 1:]))
    print(f"\nVALIDACION INTEGRAL SUPERADA: {c.passed} verificaciones.", flush=True)
    print("Listo para auditar loaders y entrenamiento E0/E1/E2.")


if __name__ == "__main__":
    try:
        validate()
    except Exception as exc:
        print(f"\nVALIDACION FALLIDA: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
