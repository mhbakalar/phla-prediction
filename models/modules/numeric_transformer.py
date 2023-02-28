import os
from torch import nn, Tensor
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from .split_transformer import SplitTransformer

class NumericTransformer(L.LightningModule):

    class SequencePooler(L.LightningModule):
        """
        Sequence pooling source from:
        https://github.com/SHI-Labs/Compact-Transformers
        """
        def __init__(self, d_model, proj_dim=1):
            super().__init__()
            self.embedding_dim = d_model
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
            self.projection = nn.Linear(self.embedding_dim, proj_dim)
                
        def forward(self, x):
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = self.projection(x)
            return x
    
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = None
        self.learning_rate = learning_rate

        # Linear network to transform results to numeric value
        self.projection = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        ## Metrics and criterion
        self.criterion = nn.MSELoss()

    def load_base_transformer(self, model: SplitTransformer):
        # Patch transformer
        self.transformer = model

        # Replace classifier
        self.transformer.classifier = self.SequencePooler(d_model=model.embedding_dim, proj_dim=1024)

    def network(self, x):
        x = self.transformer(x)
        x = self.projection(x)
        return x

    def forward(self, x):
        y = self.network(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, values = batch
        logits = self.network(inputs)
        loss = self.criterion(logits, values.unsqueeze(-1))
        self.log("bce_loss", loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(inputs)
        loss = self.criterion(logits, labels.unsqueeze(-1))

        self.log("val_loss", loss, on_epoch=True)

        return loss
    
    # Predict needs work
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.network(inputs)
        return inputs, labels.unsqueeze(-1), outputs
