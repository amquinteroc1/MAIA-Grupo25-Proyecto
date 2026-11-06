"""
cnn1d_e2.py — Modelo E2: contexto temporal con GRU.

Input:  (batch, 5, 3, 3000)  — ventana de 5 épocas consecutivas
Output: (batch, 5)           — predicción para la época central (índice 2)

Arquitectura:
    - Extractor CNN compartido (idéntico a E0/E1, pesos compartidos entre las 5 épocas)
    - Cada época → embedding de 512 valores
    - 5 embeddings procesados en orden por una GRU (hidden=256, layers=1)
    - Estado final de la GRU (256) → Linear(256 → 5) → logits

Diferencia con E1:
    - E1 concatena los 5 embeddings (2560) → Linear(2560→5)
    - E2 los pasa por una GRU que modela el orden temporal → Linear(256→5)

Proyecto: Efecto del contexto temporal en la clasificación de estadios del sueño
Grupo 25 — Universidad de los Andes, 2026
"""

import torch
import torch.nn as nn

try:
    from src.config import CNN_ARCH, E2_HEAD
except ImportError:
    from config import CNN_ARCH, E2_HEAD


class SleepCNN1D_E2(nn.Module):
    """
    CNN 1D con contexto temporal por GRU (E2).

    Recibe una ventana de 5 épocas consecutivas y clasifica la época central.
    El extractor convolucional es idéntico al de E0/E1 y sus pesos son compartidos
    entre las 5 épocas. Los embeddings se procesan en orden por una GRU unidireccional.

    Args:
        n_channels:  número de canales de señal (default: 3)
        n_classes:   número de clases de sueño (default: 5)
        dropout:     tasa de dropout en el embedding (default: 0.5)
        seq_len:     número de épocas en la ventana (default: 5)
        gru_hidden:  tamaño del estado oculto de la GRU (default: 256)
        gru_layers:  número de capas de la GRU (default: 1)
    """

    def __init__(
        self,
        n_channels:  int = CNN_ARCH["n_channels"],
        n_classes:   int = CNN_ARCH["n_classes"],
        dropout:     float = CNN_ARCH["dropout"],
        seq_len:     int = E2_HEAD["seq_len"],
        gru_hidden:  int = E2_HEAD["gru_hidden"],
        gru_layers:  int = E2_HEAD["gru_layers"],
    ):
        super().__init__()

        self.seq_len = seq_len

        # ── Extractor convolucional (idéntico a E0/E1, pesos compartidos) ────────
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

        # ── Proyección a embedding de 512 (igual al cabezal denso de E0/E1) ──────
        self.embed = nn.Sequential(
            nn.Flatten(),             # 256 * 15 = 3840
            nn.Linear(CNN_ARCH["flat_size"], CNN_ARCH["fc_hidden"]),  # 3840 → 512
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── GRU — procesa la secuencia de embeddings en orden temporal ───────────
        self.gru = nn.GRU(
            input_size=CNN_ARCH["fc_hidden"],  # 512
            hidden_size=gru_hidden,             # 256
            num_layers=gru_layers,              # 1
            batch_first=True,                   # entrada: (batch, seq, features)
            dropout=0.0,                        # sin dropout interno (seq_len corto)
        )

        # ── Cabezal de clasificación sobre el estado final de la GRU ─────────────
        self.classifier = nn.Linear(gru_hidden, n_classes)  # 256 → 5

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
            embeddings.append(emb_t.unsqueeze(1))  # (batch, 1, 512)

        # Apilar en secuencia → (batch, seq_len, 512)
        seq = torch.cat(embeddings, dim=1)

        # GRU — procesa la secuencia en orden
        # output: (batch, seq_len, gru_hidden)
        # h_n:   (num_layers, batch, gru_hidden) — estado final
        _, h_n = self.gru(seq)

        # Tomar el estado final de la última capa → (batch, gru_hidden)
        context = h_n[-1]

        # Clasificar
        return self.classifier(context)       # (batch, 5)


def build_model_e2(
    n_channels:  int = CNN_ARCH["n_channels"],
    n_classes:   int = CNN_ARCH["n_classes"],
    dropout:     float = CNN_ARCH["dropout"],
    seq_len:     int = E2_HEAD["seq_len"],
    gru_hidden:  int = E2_HEAD["gru_hidden"],
    gru_layers:  int = E2_HEAD["gru_layers"],
) -> SleepCNN1D_E2:
    """Factory function para instanciar el modelo E2."""
    return SleepCNN1D_E2(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
        seq_len=seq_len,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    )


if __name__ == "__main__":
    # Verificación rápida de shapes y parámetros
    model = build_model_e2()
    dummy = torch.randn(8, 5, 3, 3000)   # batch de 8 ventanas de 5 épocas
    out   = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")         # esperado (8, 5)
    assert out.shape == (8, 5), "ERROR: output shape incorrecto"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")

    n_features = sum(p.numel() for p in model.features.parameters())
    n_embed    = sum(p.numel() for p in model.embed.parameters())
    n_gru      = sum(p.numel() for p in model.gru.parameters())
    n_head     = sum(p.numel() for p in model.classifier.parameters())
    print(f"  → features (compartido): {n_features:,}")
    print(f"  → embed:                 {n_embed:,}")
    print(f"  → GRU:                   {n_gru:,}")
    print(f"  → classifier head:       {n_head:,}")
    print("✓ Arquitectura E2 verificada")