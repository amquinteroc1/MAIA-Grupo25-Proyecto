"""
train_cnn.py
Entrenamiento de CNN 1D para clasificación de etapas de sueño.
Registra experimento en MLflow y serializa el mejor modelo.

Uso:
    python src/models/train_cnn.py
    python src/models/train_cnn.py --epochs 50 --batch_size 128 --lr 1e-3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import mlflow
import mlflow.pytorch

from cnn1d import build_model

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR  = ROOT / "data_processed"
MODEL_DIR = ROOT / "experiments" / "cnn1d"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# ── Dataset ───────────────────────────────────────────────────────────────────
class SleepDataset(Dataset):
    def __init__(self, split: str = "train"):
        if split == "train":
            self.X = np.memmap(DATA_DIR / "X_train.npy", dtype="float32",
                               mode="r", shape=(364642, 3, 3000))
            self.y = np.load(DATA_DIR / "y_train.npy")
        else:
            self.X = np.memmap(DATA_DIR / "X_test.npy", dtype="float32",
                               mode="r", shape=(93010, 3, 3000))
            self.y = np.load(DATA_DIR / "y_test.npy")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx]))        # (3, 3000)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ── Class weights ─────────────────────────────────────────────────────────────
def calcular_class_weights(y_train: np.ndarray,
                           n_classes: int = 5) -> torch.Tensor:
    """Calcula pesos inversamente proporcionales a la frecuencia de cada clase."""
    n_total = len(y_train)
    weights = []
    for i in range(n_classes):
        n_i = (y_train == i).sum()
        w_i = n_total / (n_classes * n_i)
        weights.append(w_i)
    return torch.tensor(weights, dtype=torch.float32)


# ── Entrenamiento un epoch ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ── Evaluación ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds       = logits.argmax(1)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return total_loss / total, correct / total, f1_macro, f1_per_class, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Datos
    print("Cargando datos...")
    train_ds = SleepDataset("train")
    test_ds  = SleepDataset("test")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds):,} épocas")
    print(f"  Test:  {len(test_ds):,} épocas")

    # Class weights
    y_train      = np.load(DATA_DIR / "y_train.npy")
    class_weights = calcular_class_weights(y_train).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    # Modelo
    model     = build_model(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3,
                                   factor=0.5)

    # MLflow
    mlflow.set_experiment("sleep-stage-cnn1d")
    with mlflow.start_run(run_name=f"cnn1d-ep{args.epochs}-bs{args.batch_size}"):
        mlflow.log_params({
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "dropout":     args.dropout,
            "optimizer":   "Adam",
            "scheduler":   "ReduceLROnPlateau",
            "device":      str(device),
            "n_params":    sum(p.numel() for p in model.parameters()
                               if p.requires_grad),
        })

        best_f1   = 0.0
        best_path = MODEL_DIR / "best_cnn1d.pt"

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_dl,
                                           optimizer, criterion, device)
            va_loss, va_acc, va_f1, va_f1_cls, _, _ = evaluate(
                model, test_dl, criterion, device)
            scheduler.step(va_f1)

            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
                  f"va_loss={va_loss:.4f} va_acc={va_acc:.4f} "
                  f"va_f1={va_f1:.4f}")

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc":  tr_acc,
                "val_loss":   va_loss, "val_acc":    va_acc,
                "val_f1_macro": va_f1,
                "val_f1_wake":  float(va_f1_cls[0]),
                "val_f1_n1":    float(va_f1_cls[1]),
                "val_f1_n2":    float(va_f1_cls[2]),
                "val_f1_n3":    float(va_f1_cls[3]),
                "val_f1_rem":   float(va_f1_cls[4]),
            }, step=epoch)

            # Guardar mejor modelo
            if va_f1 > best_f1:
                best_f1 = va_f1
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_f1":      va_f1,
                    "val_acc":     va_acc,
                    "args":        vars(args),
                }, best_path)
                print(f"  ✓ Mejor modelo guardado (f1={best_f1:.4f})")

        # Reporte final
        print("\n=== Evaluación final (mejor modelo) ===")
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        _, _, f1_macro, f1_cls, preds, labels = evaluate(
            model, test_dl, criterion, device)
        print(classification_report(labels, preds,
              target_names=list(CLASES.values())))

        mlflow.log_metrics({
            "best_val_f1_macro": f1_macro,
            "best_val_f1_n1":    float(f1_cls[1]),
        })
        mlflow.pytorch.log_model(model, "cnn1d_model")
        mlflow.log_artifact(str(best_path))
        print(f"\nMejor F1 macro: {f1_macro:.4f}")
        print(f"Modelo guardado en: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--dropout",    type=float, default=0.5)
    args = parser.parse_args()
    main(args)
