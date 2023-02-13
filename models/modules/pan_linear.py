import os
from torch import optim, nn, utils, Tensor
import torch
import torch.nn as nn
import lightning as L
import math

class Net(L.LightningModule):

    def __init__(self, peptide_length: int=12, allele_length: int=60, dropout_rate: float=0.3):
        super().__init__()

        ## model parameters
        self.seq_length = peptide_length + allele_length
        self.n_amino_acids = 22
        self.embedding_dim = 32
        self.hidden_dim = 1024
        self.dropout_rate = dropout_rate

        self.embedding = torch.nn.Embedding(self.n_amino_acids, self.embedding_dim)

        ## Projection model
        peptide_aa_dims = self.embedding_dim * self.seq_length
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(peptide_aa_dims, self.hidden_dim)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def network(self, x):
        x = self.embedding(x)
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
        self.log("bce_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        pass

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(torch.flatten(inputs, 1))
        loss = self.criterion(logits, labels.unsqueeze(-1))
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(torch.flatten(inputs, 1))
        return inputs, labels.unsqueeze(-1), outputs
