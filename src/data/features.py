import numpy as np
from scipy.signal import welch


# potencia por banda
def potencia_banda(signal, fs, fmin, fmax):
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 4)))
    mask = (freqs >= fmin) & (freqs < fmax)

    if not np.any(mask):
        return np.nan

    return np.trapezoid(psd[mask], freqs[mask])


# extraer caracteristicas
def extraer_features(signal, fs):
    signal = np.asarray(signal)

    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "min": np.min(signal),
        "max": np.max(signal),
        "range": np.ptp(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "delta": potencia_banda(signal, fs, 0.5, 4),
        "theta": potencia_banda(signal, fs, 4, 8),
        "alpha": potencia_banda(signal, fs, 8, 13),
        "beta": potencia_banda(signal, fs, 13, 30)
    }