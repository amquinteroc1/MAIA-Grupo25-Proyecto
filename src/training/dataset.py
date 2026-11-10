"""
dataset.py — Clases de Dataset para E0 y E1/E2.

Carga los tensores generados por Diego desde data/processed/.
Usa memmap para los arrays X (hasta ~1.4 GB) — no los carga completos en RAM.

Reglas críticas (DATA_README.md):
- Los datos ya están normalizados. NO volver a normalizar.
- Para comparar E0 vs E1/E2 usar eligible_indices (cohorte común).
- groups_*.npy se pasa a Luigi para el análisis de Wilcoxon pareado.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from src.config import PATHS, CNN_ARCH
except ImportError:
    from config import PATHS, CNN_ARCH


# ══════════════════════════════════════════════════════════════════════════════
# E0 — épocas individuales (3, 3000)
# ══════════════════════════════════════════════════════════════════════════════

class SleepDatasetE0(Dataset):
    """
    Dataset para E0 — una época por muestra.

    Args:
        split: 'train' | 'val' | 'test' | 'st'
        eligible_only: si True, filtra a la cohorte común usando
                       eligible_indices_{split}.npy (para comparar con E1/E2).
                       Usar True en val, test y st; False en train.
    """

    def __init__(self, split: str = "train", eligible_only: bool = False):
        assert split in ("train", "val", "test", "st"), \
            f"split debe ser train/val/test/st, recibido: {split}"

        data_dir = PATHS["data_e0"]
        n_channels = CNN_ARCH["n_channels"]
        n_samples  = CNN_ARCH["n_samples"]

        # Cargar etiquetas y grupos completos en RAM (pequeños)
        self.y      = np.load(data_dir / f"y_{split}.npy")           # (N,)
        self.groups = np.load(data_dir / f"groups_{split}.npy")      # (N,)

        # X via memmap — no carga en RAM
        n_total = len(self.y)
        self._x_mm = np.load(
            str(data_dir / f"X_{split}.npy"),
            mmap_mode="r",
        )

        # Filtrar a cohorte común si se pide
        if eligible_only:
            seq_dir = PATHS["data_seq"]
            eligible = np.load(seq_dir / f"eligible_indices_{split}.npy")
            self.indices = eligible          # índices dentro del array E0
        else:
            self.indices = np.arange(n_total)

        self.split = split
        self.eligible_only = eligible_only

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        x = torch.from_numpy(np.array(self._x_mm[real_idx]))   # (3, 3000)
        y = torch.tensor(int(self.y[real_idx]), dtype=torch.long)
        return x, y

    def get_groups(self) -> np.ndarray:
        """Retorna los group IDs filtrados (para Wilcoxon de Luigi)."""
        return self.groups[self.indices]

    def get_labels(self) -> np.ndarray:
        """Retorna las etiquetas filtradas."""
        return self.y[self.indices]


# ══════════════════════════════════════════════════════════════════════════════
# E1 / E2 — secuencias de 5 épocas (5, 3, 3000)
# ══════════════════════════════════════════════════════════════════════════════

class SleepDatasetSeq(Dataset):
    """
    Dataset para E1 y E2 — ventana de 5 épocas consecutivas por muestra.
    La etiqueta corresponde a la época central (índice 2).

    Args:
        split: 'train' | 'val' | 'test' | 'st'
    """

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test", "st"), \
            f"split debe ser train/val/test/st, recibido: {split}"

        seq_dir = PATHS["data_seq"]

        # Etiquetas y grupos en RAM
        self.y      = np.load(seq_dir / f"y_seq_{split}.npy")          # (N,)
        self.groups = np.load(seq_dir / f"groups_seq_{split}.npy")     # (N,)

        # X via memmap
        self._x_mm = np.load(
            str(seq_dir / f"X_seq_{split}.npy"),
            mmap_mode="r",
        )

        self.split = split

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self._x_mm[idx]))   # (5, 3, 3000)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

    def get_groups(self) -> np.ndarray:
        return self.groups

    def get_labels(self) -> np.ndarray:
        return self.y


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y_train: np.ndarray, n_classes: int = 5) -> torch.Tensor:
    """
    Calcula pesos balanceados desde SC-train únicamente.
    Usa sklearn para seguir el mismo criterio que Diego en DATA_README.md.

    w_i = n_total / (n_classes * n_i)
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_train,
    )
    return torch.tensor(weights, dtype=torch.float32)