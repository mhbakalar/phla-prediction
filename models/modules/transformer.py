import os
from torch import optim, nn, utils, Tensor
import torch
import torch.nn as nn
import lightning as L
import math

class PeptideHLATransformer(L.LightningModule):

    class PositionalEncoding(nn.Module):

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
            super().__init__()

            self.dropout = nn.Dropout(p=dropout)
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: Tensor) -> Tensor:
            """
            Args:
                x: Tensor, shape [batch_size, seq_len, embedding_dim]
            """
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)

    class MaskedTransformerEncoder(nn.Module):

        def __init__(self, encoder_layer, num_layers):
            super().__init__()
            self.encoder_layer = encoder_layer
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
            return self.transformer_encoder.forward(x, src_key_padding_mask=src_mask)

    def __init__(self, peptide_length: int=12, allele_length: int=60, dropout_rate: float=0.3, transformer_heads: int=1, transformer_layers: int=1):
        super().__init__()

        ## Save hyperparameters to checkpoint
        self.save_hyperparameters()

        ## model parameters
        self.seq_length = peptide_length + allele_length
        self.n_amino_acids = 22

        self.embedding_dim = 32
        self.embedding = torch.nn.Embedding(self.n_amino_acids, self.embedding_dim)

        ## Transformer model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=transformer_heads,
            batch_first=True
        )
        self.transformer_encoder = self.MaskedTransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )
        self.positional_encoder = self.PositionalEncoding(d_model=self.embedding_dim, max_len=self.seq_length)

        ## Projection model
        peptide_aa_dims = self.embedding_dim * self.seq_length
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(peptide_aa_dims, peptide_aa_dims)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(peptide_aa_dims, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def network(self, x):
        src_mask = torch.empty(x.shape, device=x.device).fill_(0)
        src_mask = src_mask.masked_fill(x == 21, float('-inf'))

        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        y = torch.sigmoid(self.network(x))
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(torch.flatten(inputs, 1))
        loss = self.criterion(logits, labels.unsqueeze(-1))
        self.log("bce_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(torch.flatten(inputs, 1))
        loss = self.criterion(logits, labels.unsqueeze(-1))
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(torch.flatten(inputs, 1))
        embedding = self.embedding(inputs)
        return inputs, labels.unsqueeze(-1), outputs, embedding
