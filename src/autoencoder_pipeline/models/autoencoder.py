"""Neural network models for the GC-IMS autoencoder pipeline."""
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence indices."""

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
        seq_len = x.size(1)
        return x + self.pe[:seq_len].transpose(0, 1)


class TransformerAutoencoder(nn.Module):
    """Transformer autoencoder for 1D spectra/time-series."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dim_feedforward = dim_feedforward

        self.input_embedding = nn.Linear(1, dim_feedforward)
        self.positional_encoding = PositionalEncoding(dim_feedforward, max_len=input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.encoder_fc = nn.Linear(dim_feedforward * input_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, dim_feedforward * input_dim)

        # The latent vector is expanded into a full sequence, so reconstruction
        # only needs self-attention. A TransformerDecoder requires a separate
        # memory tensor; using the same tensor as target and memory can produce
        # repeated template-like bands instead of faithful local reconstruction.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(dim_feedforward, 1)

    def encode(self, x):
        batch = x.size(0)
        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        enc = self.transformer_encoder(x)
        flat = enc.reshape(batch, -1)
        return self.encoder_fc(flat)

    def decode(self, z):
        batch = z.size(0)
        x = self.decoder_fc(z)
        x = x.reshape(batch, self.input_dim, self.dim_feedforward)
        x = self.positional_encoding(x)
        dec = self.transformer_decoder(x)
        return self.output_layer(dec).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
