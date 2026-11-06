"""
cnn1d_e1.py — Modelo E1: contexto temporal por concatenación.

Input:  (batch, 5, 3, 3000)  — ventana de 5 épocas consecutivas
Output: (batch, 5)           — predicción para la época central (índice 2)

Arquitectura:
    - Extractor CNN compartido (idéntico a E0, pesos compartidos entre las 5 épocas)
    - Cada época → embedding de 512 valores
    - 5 embeddings concatenados → vector de 2.560
    - Linear(2560 → 5) → logits

Proyecto: Efecto del contexto temporal en la clasificación de estadios del sueño
Grupo 25 — Universidad de los Andes, 2026
"""

import torch
import torch.nn as nn

try:
    from src.config import CNN_ARCH, E1_HEAD
except ImportError:
    from config import CNN_ARCH, E1_HEAD


class SleepCNN1D_E1(nn.Module):
    """
    CNN 1D con contexto temporal por concatenación (E1).

    Recibe una ventana de 5 épocas consecutivas y clasifica la época central.
    El extractor convolucional es idéntico al de E0 y sus pesos son compartidos
    entre las 5 épocas (un solo módulo procesa cada época en el loop).

    Args:
        n_channels: número de canales de señal (default: 3)
        n_classes:  número de clases de sueño (default: 5)
        dropout:    tasa de dropout en el embedding (default: 0.5)
        seq_len:    número de épocas en la ventana (default: 5)
    """

    def __init__(
        self,
        n_channels: int = CNN_ARCH["n_channels"],
        n_classes:  int = CNN_ARCH["n_classes"],
        dropout:    float = CNN_ARCH["dropout"],
        seq_len:    int = E1_HEAD["seq_len"],
    ):
        super().__init__()

        self.seq_len = seq_len

        # ── Extractor convolucional (idéntico a E0, pesos compartidos) ──────────
        self.features = nn.Sequential(
            # Bloque 1
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

            # Bloque 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),          # 30 → 15
        )

        # ── Proyección a embedding de 512 (igual al cabezal denso de E0) ────────
        self.embed = nn.Sequential(
            nn.Flatten(),             # 256 * 15 = 3840
            nn.Linear(CNN_ARCH["flat_size"], CNN_ARCH["fc_hidden"]),  # 3840 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Cabezal de clasificación sobre los 5 embeddings concatenados ────────
        self.classifier = nn.Linear(E1_HEAD["in_features"], n_classes)  # 2560 → 5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor de forma (batch, seq_len, n_channels, 3000)
               ej. (64, 5, 3, 3000)
        Returns:
            logits de forma (batch, n_classes)
        """
        batch_size = x.size(0)

        # Procesar cada época con el extractor compartido
        embeddings = []
        for t in range(self.seq_len):
            epoch_t = x[:, t, :, :]          # (batch, 3, 3000)
            feat_t  = self.features(epoch_t)  # (batch, 256, 15)
            emb_t   = self.embed(feat_t)      # (batch, 512)
            embeddings.append(emb_t)

        # Concatenar los 5 embeddings → (batch, 2560)
        concat = torch.cat(embeddings, dim=1)

        # Clasificar
        return self.classifier(concat)        # (batch, 5)


def build_model_e1(
    n_channels: int = CNN_ARCH["n_channels"],
    n_classes:  int = CNN_ARCH["n_classes"],
    dropout:    float = CNN_ARCH["dropout"],
    seq_len:    int = E1_HEAD["seq_len"],
) -> SleepCNN1D_E1:
    """Factory function para instanciar el modelo E1."""
    return SleepCNN1D_E1(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    # Verificación rápida de shapes y parámetros
    model = build_model_e1()
    dummy = torch.randn(8, 5, 3, 3000)   # batch de 8 ventanas de 5 épocas
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")         # esperado (8, 5)
    assert out.shape == (8, 5), "ERROR: output shape incorrecto"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    # Verificar que el extractor tiene los mismos parámetros que E0
    n_features = sum(p.numel() for p in model.features.parameters())
    n_embed    = sum(p.numel() for p in model.embed.parameters())
    n_head     = sum(p.numel() for p in model.classifier.parameters())
    print(f"  → features (compartido): {n_features:,}")
    print(f"  → embed:                 {n_embed:,}")
    print(f"  → classifier head:       {n_head:,}")
    print("✓ Arquitectura E1 verificada")