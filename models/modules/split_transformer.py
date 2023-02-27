import os
from torch import nn, Tensor
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math

class Transformer(L.LightningModule):

    class TransformerClassifier(nn.Module):
        """
        Sequence pooling source from:
        https://github.com/SHI-Labs/Compact-Transformers
        """
        def __init__(self, d_model):
            super().__init__()
            self.embedding_dim = d_model
            self.attention_pool = nn.Linear(self.embedding_dim, 1)
            self.projection = nn.Linear(self.embedding_dim, 1)
                
        def forward(self, x):
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = self.projection(x)
            return x

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

    def __init__(self, peptide_length: int=12, 
                 allele_length: int=34, 
                 dropout_rate: float=0.3, 
                 embedding_dim: int=32, 
                 transformer_heads: int=1, 
                 transformer_layers: int=1,
                 learning_rate: float=1e-4):
        super().__init__()

        ## Save hyperparameters to checkpoint
        self.save_hyperparameters()

        # Manually save learning rate parameter. Compatible with auto_lr. Check MHB.
        self.learning_rate = learning_rate
        self.peptide_length = peptide_length
        self.allele_length = allele_length
        self.embedding_dim = embedding_dim

        ## model parameters
        self.seq_length = peptide_length + allele_length  # Add one for BOS
        self.n_amino_acids = 23  # Update with len(aa_map) from DataModule

        self.pep_embedding = torch.nn.Embedding(self.n_amino_acids, embedding_dim)
        self.hla_embedding = torch.nn.Embedding(self.n_amino_acids, embedding_dim)
        
        ## Peptide transformer and positional encoder
        self.pep_transformer_encoder = self.MaskedTransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.pep_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=peptide_length)
        self.pep_mask = None

        ## HLA transformer and positional encoder
        self.hla_transformer_encoder = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers        
        )
        self.hla_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=allele_length)

        ## Peptide-HLA transformer and positional encoder
        self.phla_transformer_encoder = self.MaskedTransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.phla_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=self.seq_length)
        self.phla_mask = None

        ## Projection model
        self.classifier = self.TransformerClassifier(embedding_dim)

        ## Metrics and criterion
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task='binary')

    @property
    def transformers(self):
        return {
            'pep_transformer_encoder': self.pep_transformer_encoder, 
            'hla_transformer_encoder': self.hla_transformer_encoder, 
            'phla_transformer_encoder': self.phla_transformer_encoder
        }

    def network(self, x):
        # Separate peptide and HLA from input x
        x_hla = x[:,0:self.allele_length]
        x_pep = x[:,self.allele_length:]

        # Build phla and pep masks
        if self.phla_mask is None or self.phla_mask.size(0) != len(x):
            mask_shape = (len(x), self.seq_length)
            mask = torch.empty(mask_shape, device=x.device).fill_(0)
            mask = mask.masked_fill(x == 21, float('-inf'))  # Update with blank token from aa_map
            self.phla_mask = mask
            self.pep_mask = mask[:, self.allele_length:]

        # HLA transformer network
        x_hla = self.hla_embedding(x_hla)
        x_hla = self.hla_positional_encoder(x_hla)
        x_hla = self.hla_transformer_encoder(x_hla)

        # Peptide transformer network
        x_pep = self.pep_embedding(x_pep)
        x_pep = self.pep_positional_encoder(x_pep)
        x_pep = self.pep_transformer_encoder(x_pep, self.pep_mask)
        
        # Combine peptide and HLA representations
        x_cat = torch.cat((x_pep, x_hla), dim=1)

        # Peptide transformer network
        x_cat = self.phla_positional_encoder(x_cat)
        x_cat = self.phla_transformer_encoder(x_cat, self.phla_mask)

        # Sequence pooling classifier
        x = self.classifier(x_cat)  
        return x

    def forward(self, x):
        y = torch.sigmoid(self.network(x))
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(inputs)

        loss = self.criterion(logits, labels.unsqueeze(-1))
        accuracy = self.accuracy(logits, labels.unsqueeze(-1))

        self.log("bce_loss", loss.detach(), on_epoch=True)
        self.log("accuracy", accuracy.detach(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(inputs)
        loss = self.criterion(logits, labels.unsqueeze(-1))
        
        accuracy = self.accuracy(logits, labels.unsqueeze(-1))

        self.log("val_loss", loss.detach(), on_epoch=True)
        self.log("val_accuracy", accuracy.detach(), on_epoch=True)

        return loss
    
    # Predict needs work
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(torch.flatten(inputs, 1))
        return inputs, labels.unsqueeze(-1), outputs
