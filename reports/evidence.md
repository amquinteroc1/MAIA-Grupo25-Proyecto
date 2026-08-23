# Execution evidence - Git + DVC

Date: 2026-08-23

## Remote access

- Host: `ip-172-31-15-214`
- User: `ubuntu`
- Public endpoint: `3.91.147.253:22`
- SSH authentication: successful with the project key.

## Dataset inventory

- Dataset root: `data/physionet.org/files/sleep-edfx/1.0.0`
- PSG files: 197
- Hypnogram files: 197
- Paired records: 197
- Files in `data/`: 403
- Dataset size: 8.2G
- Exploration outputs: `reports/eda/sleep_edf_summary.json` and `reports/eda/sleep_edf_records.csv`
- Channels observed include EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, submental EMG, respiration, temperature and event markers depending on the study.

## Git/DVC preparation

- Git branch: `main`
- GitHub origin configured as `https://github.com/amquinteroc1/MAIA-Grupo25-Proyecto.git`
- DVC version: 3.67.1
- DVC pointer: `data.dvc`
- Pointer size: 8,715,280,028 bytes / 403 files
- Local remote configured as `local-remote -> /tmp/dvcstore`
- Raw EDF/EDF+ files are excluded from Git by `.gitignore`.

## Pending gates

- `dvc add --to-remote` was attempted with one job and stopped with `Disk quota exceeded`; the partial, unreferenced 355M remote was removed.
- `dvc status` reports `data` not in cache, so no `dvc push`, `dvc pull`, or second data version is claimed.
- S3 is intentionally pending a real bucket and AWS Academy credentials.
- Git commit/push is pending a real contributor identity and interactive HTTPS PAT; no token was stored.
