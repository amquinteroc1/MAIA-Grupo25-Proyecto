#!/usr/bin/env python3
"""Auditoria y limpieza conservadora del repositorio MAIA Grupo 25.

Ubicacion: raiz del repositorio, junto a .gitignore y subir_HF.py.

    python mantenimiento_repo_tesis.py
    python mantenimiento_repo_tesis.py --apply
    python mantenimiento_repo_tesis.py --apply --clean-caches

Nunca borra datos, objetos DVC, checkpoints, predicciones, metricas ni imagenes.
No hace git add/commit/push ni dvc push/gc.
"""
from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
from pathlib import Path

BEGIN = "<!-- INICIO: TESIS E0-E1-E2 2026 -->"
END = "<!-- FIN: TESIS E0-E1-E2 2026 -->"

CURRENT = '''{begin}
## Tesis 2026: efecto del contexto temporal en la clasificación de estadios del sueño

**Estado:** entrenamiento y evaluación finalizados para E0, E1 y E2; resultados externos en
Sleep-Telemetry (ST). Los experimentos anteriores SVM/LightGBM/CNN1D se mantienen como
antecedentes del microproyecto y **no son directamente comparables** con estas métricas:
se utilizaron particiónes y condiciones de evaluación diferentes.

### Datos y protocolo experimental actual

- Fuente: PhysioNet Sleep-EDF Expanded, versiónes SC (Sleep-Cassette) y ST (Sleep-Telemetry).
- Entrada: EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal; 100 Hz; épocas de 30 segundos.
- Etiquetas: Wake, N1, N2, N3 y REM.
- División por sujeto SC: entrenamiento 49, validación 13, prueba 16; ST externo 22 sujetos.
- Evaluación SC-test: **39.961 épocas** comúnes para las tres arquitecturas.
- Evaluación ST: **42.003 épocas** comúnes para las tres arquitecturas.
- Semillas por modelo: `42`, `123` y `2026` (nueve entrenamientos independientes).

### Arquitecturas

| Modelo | Entrada y agregación temporal |
|---|---|
| E0 | CNN 1D sobre una epoca de 30 s, sin contexto |
| E1 | CNN compartida sobre 5 épocas y concatenacion de embeddings |
| E2 | CNN compartida sobre 5 épocas y una GRU para agregación temporal |

La ventana objetivo en E1 y E2 es la epoca central de las cinco consecutivas.
El extractor CNN tiene cuatro bloques convolucionales, con 32/64/128/256 filtros.
Los tres modelos usan un protocolo de entrenamiento común; consultar `src/config.py`.

### Resultados: promedio de tres semillas

| Modelo | Accuracy SC-test | F1-macro SC-test | Accuracy ST | F1-macro ST | F1 N1 ST |
|---|---:|---:|---:|---:|---:|
| E0 | 78,36 % | 0,7601 | 65,26 % | 0,6090 | 0,3603 |
| E1 | 81,51 % | 0,7814 | 61,19 % | 0,5893 | 0,3282 |
| **E2** | **82,34 %** | **0,7888** | **67,81 %** | **0,6354** | **0,3884** |

En SC-test, E1 y E2 superan a E0. En ST, E2 mantiene el mejor F1-macro,
mientras E1 obtiene un resultado inferior a E0. Son medias descriptivas;
no constituyen una prueba de significacion estadistica por sujeto.

### Comandos de reproducibilidad de la tesis

Ejecutar desde la raiz del repositorio, con las dependencias ya instaladas:

```powershell
# Revisar la partición y los datos preparados
python -m src.data_tesis.test_data_tesis

# Regenerar datos de tesis cuando se disponga de los EDF originales
python -m src.data_tesis.crear_dataset_tesis --fase todo

# Comprobar configuración y particiónes sin entrenar
python -m src.training.train --experiment E0 --seed 42 --dry-run

# Entrenar las nueve corridas (solo si es necesario)
python -m src.training.train --experiment all --seed all

# Recalcular evaluación ST y matrices desde predicciones ya guardadas
python -m src.evaluation.evaluacion_st
python -m src.evaluation.generar_matrices_confusion

# Inventario para una futura publicación en Hugging Face (sin subida)
python subir_HF.py --root . --folder revision-luigi --dry-run
```

Los artefactos locales se guardan en `experiments/E0/`, `experiments/E1/`,
`experiments/E2/`, `outputs/metrics/` y `src/evaluation/imagenes/`.
No es necesario reentrenar para reproducir las métricas a partir de las predicciones guardadas.

### Versionado de datos y artefactos

- `data.dvc` registra el directorio `data/` (399 archivos y 8,72 GB en el metadato
  versiónado de Git). El remote DVC configurado es `cnn-remote` en S3. Su
  sincronización **debe verificarse** con `dvc status -c` antes de hacer limpieza.
- Los checkpoints, matrices y predicciones de los nueve experimentos permanecen
  locales hasta que se confirme su copia en Hugging Face u otro almacenamiento.
- Git versióna código y metadatos, pero **no sustituye** `dvc push` ni el respaldo
  de los pesos.
- Los repositorios Hugging Face originales son históricos; los repositorios `-v2`
  todavía requieren creacion/permisos antes de documentarlos como disponibles.

### Componentes anteriores

Las secciones siguientes documentan el **microproyecto previo** (SVM, LightGBM,
CNN1D, API, dashboard y despliegue). Se conservan como referencia, pero sus
comandos, rutas y resultados pueden corresponder a una versión distinta de la tesis.
{end}'''.format(begin=BEGIN, end=END)

CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SCAN_ROOTS = ("src", "tests", "dashboard", "experiments")


def git_dirty(root: Path, rel: str) -> bool:
    try:
        cp = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=root, capture_output=True, text=True, check=True,
        )
        return bool(cp.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return True


def update_readme(text: str) -> str:
    if BEGIN in text:
        if END not in text:
            raise ValueError("README tiene marcador de inicio, pero no de fin")
        i = text.index(BEGIN)
        j = text.index(END) + len(END)
        return text[:i] + CURRENT + text[j:]
    # Conserva el README historico completo y antepone informacion vigente.
    parts = text.split("\n", 1)
    if len(parts) == 1:
        return text.rstrip() + "\n\n" + CURRENT + "\n"
    return parts[0] + "\n\n" + CURRENT + "\n\n" + parts[1].lstrip("\n")


def update_train(text: str) -> str:
    # Solo referencias documentales: nunca modificar la logica del entrenamiento.
    text = text.replace("Ubicacion: src/training/train_tesis.py", "Ubicacion: src/training/train.py")
    text = text.replace("python -m src.training.train_tesis", "python -m src.training.train")
    return text


def update_hf(text: str) -> str:
    text = text.replace(
        "python -m src.evaluation.subir_huggingface_tesis --dry-run",
        "python subir_HF.py --folder revision-luigi --dry-run",
    ).replace(
        "python -m src.evaluation.subir_huggingface_tesis\n",
        "python subir_HF.py --folder revision-luigi\n",
    )
    text = text.replace(
        'default=Path(__file__).resolve().parents[2],',
        'default=Path(__file__).resolve().parent,',
    ).replace(
        'help="Raiz del repositorio local (por defecto, dos niveles sobre src/evaluation)"',
        'help="Raiz del repositorio local (por defecto, carpeta que contiene subir_HF.py)"',
    )
    return text


def update_gitignore(text: str) -> str:
    additions = ["/_hf_stage_*/", ".pytest_cache/", ".ruff_cache/", ".mypy_cache/"]
    lines = set(text.splitlines())
    new = [s for s in additions if s not in lines]
    if new:
        return text.rstrip() + "\n\n# Archivos temporales regenerables\n" + "\n".join(new) + "\n"
    return text


def cache_dirs(root: Path) -> list[Path]:
    found = []
    for name in SCAN_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_dir() and p.name in CACHE_NAMES and not any(q in {".venv", "venv", ".git", ".dvc"} for q in p.relative_to(root).parts):
                found.append(p)
    return sorted(set(found))


def bytes_in(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="Aplica solo las correcciones seguras")
    ap.add_argument("--clean-caches", action="store_true", help="Incluye borrar caches Python regenerables")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    if not (root / ".git").exists():
        ap.error("Guarda este script en la raiz del repositorio Git")
    checks = {
        "README.md": update_readme,
        "src/training/train.py": update_train,
        "subir_HF.py": update_hf,
        ".gitignore": update_gitignore,
    }
    changes = {}
    for rel, func in checks.items():
        path = root / rel
        if not path.exists():
            print(f"AVISO: no existe {rel}; se omite")
            continue
        old = path.read_text(encoding="utf-8-sig")
        new = func(old)
        if old != new:
            changes[rel] = (old, new)
            print(f"ACTUALIZAR: {rel}")
        else:
            print(f"SIN CAMBIOS: {rel}")
    print("\nDATOS PROTEGIDOS (no borrar):")
    for rel in ("data", "data.dvc", "experiments/E0", "experiments/E1", "experiments/E2", "mlruns.db", "src/evaluation/imagenes", ".dvc"):
        if (root / rel).exists():
            print(f"  {rel}")
    print("\nCODIGO HEREDADO PARA REVISAR (no se borra automaticamente):")
    legacy = (
        "experiments/experiment_svm.py",
        "experiments/experiment_lightgbm.py",
        "src/data/crear_dataset.py",
        "src/data/crear_dataset_lightgbm.py",
        "src/models/train_cnn.py",
        "notebooks/train_cnn1d.ipynb",
    )
    existing = [rel for rel in legacy if (root / rel).exists()]
    if existing:
        for rel in existing:
            print("  ", rel)
    else:
        print("  No se encontraron archivos de esta lista")
    caches = cache_dirs(root)
    total = sum(bytes_in(p) for p in caches)
    print(f"\nCaches regenerables detectadas: {len(caches)} directorios, {total / 1024**2:.2f} MiB")
    for p in caches[:20]:
        print("  ", p.relative_to(root))
    if len(caches) > 20:
        print(f"  ... y {len(caches)-20} mas")
    print("\nPENDIENTE DE REVISION (nunca se modifica aqui): git diff -- .dvc/.gitignore")
    print("PENDIENTE DE DVC: dvc status; dvc status -c; dvc remote list")
    if not args.apply:
        print("\nMODO SIMULACION: nada ha cambiado. Usa --apply para actualizar documentacion.")
        if caches:
            print("Para limpiar caches: --apply --clean-caches")
        return 0
    conflicts = [p for p in changes if git_dirty(root, p)]
    if conflicts:
        print("\nERROR: estos archivos tienen cambios Git no confirmados; no se sobrescriben:")
        for p in conflicts:
            print("  ", p)
        print("Revisa con git diff antes de ejecutar --apply. No se modifico ningun archivo.")
        return 2
    for rel, (old, new) in changes.items():
        path = root / rel
        # Respeta el tipo de salto de linea previo para evitar diffs masivos en Windows.
        original = path.read_bytes()
        newline = "\r\n" if b"\r\n" in original else "\n"
        data = new.replace("\r\n", "\n").replace("\n", newline).encode("utf-8")
        path.write_bytes(data)
        print("ACTUALIZADO:", rel)
    if args.clean_caches:
        for p in caches:
            if p.exists():
                shutil.rmtree(p)
        print(f"CACHE ELIMINADA: {len(caches)} directorios regenerables")
    print("\nLISTO. Revisa: git diff --stat; git diff --check; git status --short")
    print("No se hizo commit ni push. No se modifico DVC ni Hugging Face.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
