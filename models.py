"""
Neural network models for GC-IMS pipeline.
"""
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Injects sequence index info into transformer inputs.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].transpose(0, 1)


class TransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder for 1D timeseries.

    Encoder:
        Input (batch, seq_len) → Linear → TransformerEncoder → Flatten → FC → latent

    Decoder:
        latent → FC_expand → (batch, seq_len, dim_feedforward)
               → TransformerDecoder → Linear → (batch, seq_len)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_feedforward

        # === Input embedding ===
        self.input_embedding = nn.Linear(1, dim_feedforward)
        self.positional_encoding = PositionalEncoding(dim_feedforward, max_len=input_dim)

        # === Encoder transformer ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Latent bottleneck ===
        self.encoder_fc = nn.Linear(dim_feedforward * input_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, dim_feedforward * input_dim)

        # === Decoder transformer ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_layer = nn.Linear(dim_feedforward, 1)

    def encode(self, x):
        batch = x.size(0)

        x = x.unsqueeze(-1)  # → (batch, seq_len, 1)
        x = self.input_embedding(x)  # → (batch, seq_len, dim_feedforward)

        x = self.positional_encoding(x)

        enc = self.transformer_encoder(x)  # → (batch, seq_len, dim_feedforward)

        flat = enc.reshape(batch, -1)  # → (batch, seq_len * dim_feedforward)
        latent = self.encoder_fc(flat)  # → (batch, latent_dim)
        return latent

    def decode(self, z):
        batch = z.size(0)

        x = self.decoder_fc(z)  # → (batch, seq_len*dim_feedforward)
        x = x.reshape(batch, self.input_dim, self.dim_feedforward)

        # Using self-memory for now (auto-regressive dec not needed)
        dec = self.transformer_decoder(x, x)

        out = self.output_layer(dec).squeeze(-1)  # → (batch, seq_len)
        return out

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z