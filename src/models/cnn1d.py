"""
CNN 1D para clasificación de etapas de sueño.
Input:  (batch, 3, 3000)  — 3 canales EEG/EOG, 3000 muestras (30s @ 100Hz)
Output: (batch, 5)        — Wake, N1, N2, N3, REM
"""

import torch
import torch.nn as nn


class SleepCNN1D(nn.Module):
    """
    CNN 1D de 4 bloques convolucionales para clasificación de etapas de sueño.
    
    Arquitectura:
        Bloque 1: Conv(3→32,  k=7) + BN + ReLU + MaxPool(5)  → (32, 600)
        Bloque 2: Conv(32→64, k=5) + BN + ReLU + MaxPool(5)  → (64, 120)
        Bloque 3: Conv(64→128,k=3) + BN + ReLU + MaxPool(4)  → (128, 30)
        Bloque 4: Conv(128→256,k=3)+ BN + ReLU + MaxPool(2)  → (256, 15)
        Flatten  → FC(3840→512) + Dropout(0.5) → FC(512→5)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 5,
                 dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1 — kernels grandes para patrones de baja frecuencia
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(5),          # 3000 → 600

            # Bloque 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(5),          # 600 → 120

            # Bloque 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),          # 120 → 30

            # Bloque 4 — kernels pequeños para patrones locales
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 30 → 15
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),             # 256 * 15 = 3840
            nn.Linear(3840, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor de forma (batch, n_channels, 3000)
        Returns:
            logits de forma (batch, n_classes)
        """
        x = self.features(x)
        return self.classifier(x)


def build_model(n_channels: int = 3, n_classes: int = 5,
                dropout: float = 0.5) -> SleepCNN1D:
    """Factory function para instanciar el modelo."""
    return SleepCNN1D(n_channels=n_channels,
                      n_classes=n_classes,
                      dropout=dropout)


if __name__ == '__main__':
    # Verificación rápida de shapes
    model = build_model()
    dummy = torch.randn(8, 3, 3000)   # batch de 8 épocas
    out   = model(dummy)
    print(f'Input:  {dummy.shape}')
    print(f'Output: {out.shape}')     # esperado (8, 5)
    assert out.shape == (8, 5), 'ERROR: output shape incorrecto'
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parámetros entrenables: {n_params:,}')
    print('✓ Arquitectura verificada')
