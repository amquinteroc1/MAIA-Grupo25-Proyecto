"""
trainer.py — Loop de entrenamiento con early stopping, MLflow y artefactos.

Genera para cada corrida:
  - Checkpoint del mejor modelo (.pt)
  - Curvas de entrenamiento (loss y métricas por época)
  - Predicciones sobre SC-test y ST (cohorte común) con grupos para Luigi
  - Métricas finales completas
  - Costo computacional (parámetros, tiempos, memoria)
"""

import time
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

try:
    from src.config import TRAINING, CLASS_NAMES, N_CLASSES, MLFLOW
except ImportError:
    from config import TRAINING, CLASS_NAMES, N_CLASSES, MLFLOW


# ══════════════════════════════════════════════════════════════════════════════
# LOOP DE UNA ÉPOCA
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Entrena una época. Retorna (loss_promedio, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    """
    Evalúa el modelo sobre un DataLoader.
    Retorna métricas completas + predicciones + probabilidades.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            logits = model(x_batch)
            loss   = criterion(logits, y_batch)

            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)

            total_loss += loss.item() * len(y_batch)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_proba = np.array(all_probs)   # (N, 5)

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "loss":              total_loss / total,
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":             cohen_kappa_score(y_true, y_pred),
        "f1_per_class":      f1_per_class,       # array (5,)
        "y_true":            y_true,
        "y_pred":            y_pred,
        "y_proba":           y_proba,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    experiment: str,
    seed: int,
    run_dir: Path,
    groups_test: np.ndarray,
    st_loader: Optional[DataLoader] = None,
    groups_st: Optional[np.ndarray] = None,
) -> Dict:
    """
    Bucle principal de entrenamiento con:
    - Early stopping sobre val_f1_macro
    - MLflow logging por época
    - Guardado del mejor checkpoint
    - Artefactos de predicciones para Luigi

    Retorna dict con métricas finales y rutas de artefactos.
    """
    max_epochs   = TRAINING["max_epochs"]
    es_patience  = TRAINING["early_stopping_patience"]
    monitor      = TRAINING["monitor_metric"]   # "val_f1_macro"

    run_name = MLFLOW["run_name_template"].format(experiment=experiment, seed=seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"best_model_{experiment}_seed{seed}.pt"

    best_metric   = -1.0
    best_epoch    = 0
    epochs_no_imp = 0
    history       = []

    # ── Tiempo de entrenamiento ───────────────────────────────────────────────
    t_train_start = time.perf_counter()

    mlflow.set_experiment(MLFLOW["experiment_name"])
    with mlflow.start_run(run_name=run_name) as run:

        # Registrar parámetros
        mlflow.log_params({
            "experiment":              experiment,
            "seed":                    seed,
            "batch_size":              TRAINING["batch_size"],
            "lr":                      TRAINING["lr"],
            "weight_decay":            TRAINING["weight_decay"],
            "max_epochs":              max_epochs,
            "early_stopping_patience": es_patience,
            "scheduler_patience":      TRAINING["scheduler_patience"],
            "scheduler_factor":        TRAINING["scheduler_factor"],
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device":   str(device),
        })

        # ── Loop de épocas ────────────────────────────────────────────────────
        for epoch in range(1, max_epochs + 1):
            t_ep = time.perf_counter()

            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_res         = evaluate(model, val_loader, criterion, device)

            val_f1     = val_res["f1_macro"]
            val_f1_cls = val_res["f1_per_class"]

            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            ep_time    = time.perf_counter() - t_ep

            # Logging MLflow por época
            mlflow.log_metrics({
                "train_loss":        tr_loss,
                "train_acc":         tr_acc,
                "val_loss":          val_res["loss"],
                "val_acc":           val_res["accuracy"],
                "val_balanced_acc":  val_res["balanced_accuracy"],
                "val_f1_macro":      val_f1,
                "val_kappa":         val_res["kappa"],
                "val_f1_wake":       float(val_f1_cls[0]),
                "val_f1_n1":         float(val_f1_cls[1]),
                "val_f1_n2":         float(val_f1_cls[2]),
                "val_f1_n3":         float(val_f1_cls[3]),
                "val_f1_rem":        float(val_f1_cls[4]),
                "lr":                current_lr,
            }, step=epoch)

            history.append({
                "epoch":    epoch,
                "tr_loss":  tr_loss,
                "tr_acc":   tr_acc,
                "val_loss": val_res["loss"],
                "val_f1":   val_f1,
                "val_f1_n1": float(val_f1_cls[1]),
            })

            print(
                f"[{experiment} seed={seed}] "
                f"ep {epoch:3d}/{max_epochs} | "
                f"tr_loss={tr_loss:.4f} | "
                f"val_f1={val_f1:.4f} (N1={val_f1_cls[1]:.3f}) | "
                f"lr={current_lr:.2e} | {ep_time:.1f}s"
            )

            # ── Early stopping ────────────────────────────────────────────────
            if val_f1 > best_metric:
                best_metric   = val_f1
                best_epoch    = epoch
                epochs_no_imp = 0
                torch.save({
                    "epoch":        epoch,
                    "experiment":   experiment,
                    "seed":         seed,
                    "model_state":  model.state_dict(),
                    "val_f1_macro": val_f1,
                    "val_accuracy": val_res["accuracy"],
                }, checkpoint_path)
                print(f"  ✓ Mejor modelo guardado — val_f1_macro={best_metric:.4f}")
            else:
                epochs_no_imp += 1
                if epochs_no_imp >= es_patience:
                    print(f"  Early stopping en época {epoch} (sin mejora por {es_patience} épocas)")
                    break

        t_train_total = time.perf_counter() - t_train_start

        # ── Evaluación final con mejor modelo ─────────────────────────────────
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        print(f"\n=== Evaluación final — mejor modelo (época {best_epoch}) ===")

        # SC-test (cohorte común)
        t_inf_start = time.perf_counter()
        test_res    = evaluate(model, test_loader, criterion, device)
        t_inf_test  = time.perf_counter() - t_inf_start

        print(classification_report(
            test_res["y_true"], test_res["y_pred"],
            target_names=CLASS_NAMES, digits=4,
        ))

        # ST (si está disponible — Luigi lo usará, guardamos las predicciones)
        st_res = None
        if st_loader is not None:
            t_inf_st_start = time.perf_counter()
            st_res = evaluate(model, st_loader, criterion, device)
            t_inf_st = time.perf_counter() - t_inf_st_start

        # ── Memoria GPU ───────────────────────────────────────────────────────
        gpu_mem_mb = 0
        if device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # ── Guardar artefactos para Luigi ──────────────────────────────────────
        # SC-test
        np.save(run_dir / f"y_true_test_{experiment}_seed{seed}.npy",  test_res["y_true"])
        np.save(run_dir / f"y_pred_test_{experiment}_seed{seed}.npy",  test_res["y_pred"])
        np.save(run_dir / f"y_proba_test_{experiment}_seed{seed}.npy", test_res["y_proba"])
        np.save(run_dir / f"groups_test_{experiment}_seed{seed}.npy",  groups_test)

        # ST
        if st_res is not None and groups_st is not None:
            np.save(run_dir / f"y_true_st_{experiment}_seed{seed}.npy",  st_res["y_true"])
            np.save(run_dir / f"y_pred_st_{experiment}_seed{seed}.npy",  st_res["y_pred"])
            np.save(run_dir / f"y_proba_st_{experiment}_seed{seed}.npy", st_res["y_proba"])
            np.save(run_dir / f"groups_st_{experiment}_seed{seed}.npy",  groups_st)

        # Curvas de aprendizaje
        np.save(run_dir / f"history_{experiment}_seed{seed}.npy", history)

        # Métricas finales en MLflow
        test_f1_cls = test_res["f1_per_class"]
        final_metrics = {
            "test_accuracy":          test_res["accuracy"],
            "test_balanced_accuracy": test_res["balanced_accuracy"],
            "test_f1_macro":          test_res["f1_macro"],
            "test_kappa":             test_res["kappa"],
            "test_f1_wake":           float(test_f1_cls[0]),
            "test_f1_n1":             float(test_f1_cls[1]),
            "test_f1_n2":             float(test_f1_cls[2]),
            "test_f1_n3":             float(test_f1_cls[3]),
            "test_f1_rem":            float(test_f1_cls[4]),
            "best_epoch":             float(best_epoch),
            "train_time_s":           t_train_total,
            "inference_time_test_s":  t_inf_test,
            "gpu_mem_peak_mb":        gpu_mem_mb,
            "n_params": float(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(str(checkpoint_path))

        run_id = run.info.run_id
        print(f"\nMLflow run_id: {run_id}")
        print(f"Tiempo de entrenamiento: {t_train_total:.1f}s")
        print(f"Tiempo de inferencia SC-test: {t_inf_test:.2f}s")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_id":       run_id,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_metric,
        "test_metrics": final_metrics,
        "history":      history,
        "run_dir":      run_dir,
    }