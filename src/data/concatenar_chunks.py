"""
concatenar_chunks.py
Concatena 197 archivos .npz a disco usando memmap — sin cargar todo en RAM.
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json

tmp_dir = Path("data_processed/tmp")
out_dir = Path("data_processed")

archivos = sorted(tmp_dir.glob("*.npz"))
print(f"Archivos encontrados: {len(archivos)}")

# ── Paso 1: calcular tamaño total sin cargar datos ────────────────────────────
print("Calculando tamaño total...")
total_epocas = 0
for f in archivos:
    d = np.load(f)
    total_epocas += d["X"].shape[0]
print(f"Total épocas: {total_epocas}")

# ── Paso 2: crear arrays en disco con memmap ──────────────────────────────────
X_all = np.memmap(out_dir / "X_all.mmap", dtype="float32",
                  mode="w+", shape=(total_epocas, 3, 3000))
y_all = np.memmap(out_dir / "y_all.mmap", dtype="int64",
                  mode="w+", shape=(total_epocas,))
g_all = np.memmap(out_dir / "g_all.mmap", dtype="int64",
                  mode="w+", shape=(total_epocas,))

# ── Paso 3: escribir sujeto por sujeto ────────────────────────────────────────
cursor = 0
for i, f in enumerate(archivos):
    d = np.load(f)
    n = d["X"].shape[0]
    X_all[cursor:cursor+n] = d["X"]
    y_all[cursor:cursor+n] = d["y"]
    g_all[cursor:cursor+n] = d["g"]
    cursor += n
    print(f"[{i+1}/{len(archivos)}] {f.name} — {n} épocas acumuladas: {cursor}")

print(f"✓ Total escrito: {cursor} épocas")
print(f"Distribución: { {i: int((y_all==i).sum()) for i in range(5)} }")

# ── Paso 4: split por sujeto ──────────────────────────────────────────────────
print("Aplicando split 80/20 por sujeto...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
idx_train, idx_test = next(gss.split(X_all, y_all, groups=g_all))

y_train = np.array(y_all[idx_train])
y_test  = np.array(y_all[idx_test])
g_train = np.array(g_all[idx_train])
g_test  = np.array(g_all[idx_test])

assert len(set(g_train.tolist()) & set(g_test.tolist())) == 0
print(f"Train: {len(idx_train)} épocas | Test: {len(idx_test)} épocas")
print("✓ Split verificado — sin data leakage")


# ── Paso 5: guardar artefactos finales en bloques ────────────────────────────
print("Guardando artefactos finales en bloques...")
BLOCK = 1000

# X_train
X_train_out = np.memmap(out_dir / "X_train.npy", dtype="float32",
                        mode="w+", shape=(len(idx_train), 3, 3000))
for i in range(0, len(idx_train), BLOCK):
    bloque = idx_train[i:i+BLOCK]
    X_train_out[i:i+len(bloque)] = X_all[bloque]
    if i % 50000 == 0:
        print(f"  X_train: {i}/{len(idx_train)}")
del X_train_out
print("✓ X_train guardado")

# X_test
X_test_out = np.memmap(out_dir / "X_test.npy", dtype="float32",
                       mode="w+", shape=(len(idx_test), 3, 3000))
for i in range(0, len(idx_test), BLOCK):
    bloque = idx_test[i:i+BLOCK]
    X_test_out[i:i+len(bloque)] = X_all[bloque]
    if i % 20000 == 0:
        print(f"  X_test: {i}/{len(idx_test)}")
del X_test_out
print("✓ X_test guardado")

# ── Paso 6: metadata ──────────────────────────────────────────────────────────
split_info = {
    "n_epocas_train":     int(len(y_train)),
    "n_epocas_test":      int(len(y_test)),
    "n_sujetos_train":    len(set(g_train.tolist())),
    "n_sujetos_test":     len(set(g_test.tolist())),
    "distribucion_train": {str(i): int((y_train==i).sum()) for i in range(5)},
    "distribucion_test":  {str(i): int((y_test==i).sum())  for i in range(5)},
    "clases": {"0":"Wake","1":"N1","2":"N2","3":"N3","4":"REM"}
}
with open(out_dir / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)
print("✓ split_info.json guardado")
print("=== Concatenación completada ===")