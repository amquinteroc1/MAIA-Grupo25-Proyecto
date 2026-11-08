"""Entrenamiento E0/E1/E2 con los datos corregidos de Sleep-EDF.

Ubicacion: src/training/train_tesis.py
Uso desde la raiz del proyecto:
  python -m src.training.train_tesis --experiment E0 --seed 42 --dry-run
  python -m src.training.train_tesis --experiment E0 --seed 42
  python -m src.training.train_tesis --experiment all --seed all
"""

import argparse
import gc
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import CNN_ARCH, E1_HEAD, E2_HEAD, MLFLOW, PATHS, SEEDS, TRAINING
from src.models.cnn1d import SleepCNN1D
from src.models.cnn1d_e1 import SleepCNN1D_E1
from src.models.cnn1d_e2 import SleepCNN1D_E2
from src.training.dataset import SleepDatasetE0, SleepDatasetSeq, compute_class_weights
from src.training.trainer import run_training


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_datasets(experiment):
    if experiment == "E0":
        return {
            "train": SleepDatasetE0("train", eligible_only=False),
            "val": SleepDatasetE0("val", eligible_only=True),
            "test": SleepDatasetE0("test", eligible_only=True),
            "st": SleepDatasetE0("st", eligible_only=True),
        }
    return {split: SleepDatasetSeq(split) for split in ("train", "val", "test", "st")}


def build_model(experiment):
    common = {
        "n_channels": CNN_ARCH["n_channels"],
        "n_classes": CNN_ARCH["n_classes"],
        "dropout": CNN_ARCH["dropout"],
    }
    if experiment == "E0":
        return SleepCNN1D(**common)
    if experiment == "E1":
        return SleepCNN1D_E1(**common, seq_len=E1_HEAD["seq_len"])
    return SleepCNN1D_E2(
        **common,
        seq_len=E2_HEAD["seq_len"],
        gru_hidden=E2_HEAD["gru_hidden"],
        gru_layers=E2_HEAD["gru_layers"],
    )


def validate_datasets(experiment, data):
    for split, dataset in data.items():
        assert len(dataset) > 0, f"Dataset vacio: {experiment}/{split}"
        print(f"  {split:5s}: {len(dataset):,} muestras")

    # Los sujetos deben ser exclusivos por particion.
    groups = {s: set(map(str, ds.get_groups())) for s, ds in data.items()}
    splits = list(groups)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert groups[a].isdisjoint(groups[b]), f"Fuga de sujetos: {a}/{b}"

    # E0 se evalua exactamente en los mismos centros que E1/E2.
    if experiment == "E0":
        for split in ("val", "test", "st"):
            seq = SleepDatasetSeq(split)
            assert np.array_equal(data[split].get_labels(), seq.get_labels()), split
            assert np.array_equal(data[split].get_groups(), seq.get_groups()), split
            del seq


def make_loader(data, split, seed, device):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        data[split],
        batch_size=TRAINING["batch_size"],
        shuffle=(split == "train"),
        num_workers=TRAINING["num_workers"],
        pin_memory=(TRAINING["pin_memory"] and device.type == "cuda"),
        generator=gen,
    )


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"No serializable: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Entrenar E0, E1 y E2 para la tesis")
    parser.add_argument("--experiment", choices=("E0", "E1", "E2", "all"), default="E0")
    parser.add_argument("--seed", choices=("42", "123", "2026", "all"), default="42")
    parser.add_argument("--dry-run", action="store_true", help="Solo verificar, sin entrenar")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    assert (root / "src" / "config.py").is_file(), "Ubica el script en src/training/"
    assert TRAINING["monitor_metric"] == "val_f1_macro"

    experiments = ("E0", "E1", "E2") if args.experiment == "all" else (args.experiment,)
    seeds = tuple(SEEDS) if args.seed == "all" else (int(args.seed),)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.dry_run:
        raise RuntimeError("CUDA no disponible: entrenamiento detenido")

    mlflow.set_tracking_uri(MLFLOW["tracking_uri"])
    print("Dispositivo:", device)
    print("Batch size:", TRAINING["batch_size"])
    print("Max epochs:", TRAINING["max_epochs"])
    print("Early stopping:", TRAINING["early_stopping_patience"])
    print("Plan:", [(e, s) for e in experiments for s in seeds], flush=True)

    # Evitar sobrescribir checkpoints/resultados, incluso en ejecuciones multiples.
    for exp in experiments:
        for seed in seeds:
            folder = PATHS["experiments"] / exp / f"seed{seed}"
            ckpt = folder / f"best_model_{exp}_seed{seed}.pt"
            if ckpt.exists() or (folder / "run_summary.json").exists():
                raise FileExistsError(f"Ya existen resultados: {folder}")

    for exp in experiments:
        print(f"\nDataset {exp}:")
        data = get_datasets(exp)
        validate_datasets(exp, data)
        if args.dry_run:
            print(f"{exp}: verificacion correcta (sin entrenamiento)")
            del data
            continue

        for seed in seeds:
            fix_seed(seed)
            folder = PATHS["experiments"] / exp / f"seed{seed}"
            print(f"\nINICIANDO {exp} / semilla {seed}", flush=True)
            torch.cuda.reset_peak_memory_stats()

            model = build_model(exp).to(device)
            weights = compute_class_weights(data["train"].get_labels(), n_classes=5).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=TRAINING["lr"],
                weight_decay=TRAINING["weight_decay"],
            )
            scheduler = ReduceLROnPlateau(
                optimizer, mode="max",
                patience=TRAINING["scheduler_patience"],
                factor=TRAINING["scheduler_factor"],
                min_lr=TRAINING["scheduler_min_lr"],
            )
            loaders = {s: make_loader(data, s, seed, device) for s in data}

            result = run_training(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                test_loader=loaders["test"],
                st_loader=loaders["st"],
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                experiment=exp,
                seed=seed,
                run_dir=folder,
                groups_test=data["test"].get_groups(),
                groups_st=data["st"].get_groups(),
            )

            summary = {
                "experiment": exp,
                "seed": seed,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_sizes": {s: len(d) for s, d in data.items()},
                "training_config": dict(TRAINING),
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_val_f1"],
                "mlflow_run_id": result["run_id"],
                "test_metrics": result["test_metrics"],
            }
            with (folder / "run_summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=json_default)
            print(f"COMPLETADO {exp} seed={seed}: {folder}", flush=True)

            del model, weights, criterion, optimizer, scheduler, loaders, result
            gc.collect()
            torch.cuda.empty_cache()
        del data
        gc.collect()

    print("\nVERIFICACION COMPLETA" if args.dry_run else "\nENTRENAMIENTOS COMPLETADOS")


if __name__ == "__main__":
    main()
