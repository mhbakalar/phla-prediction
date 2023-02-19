import os
from torch import nn, Tensor
import torch
import torchmetrics
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

    def __init__(self, peptide_length: int=12, 
                 allele_length: int=60, 
                 dropout_rate: float=0.3, 
                 embedding_dim: int=32, 
                 transformer_heads: int=1, 
                 transformer_layers: int=1,
                 learning_rate: float=1e-3):
        super().__init__()

        ## Save hyperparameters to checkpoint
        self.save_hyperparameters()

        # Manually save learning rate parameter. Compatible with auto_lr. Check MHB.
        self.learning_rate = learning_rate

        ## model parameters
        self.seq_length = 1 + peptide_length + allele_length  # Add one for BOS
        self.n_amino_acids = 23  # Update with len(aa_map) from DataModule

        self.bos_embedding = torch.nn.Embedding(self.n_amino_acids, embedding_dim)
        self.pep_embedding = torch.nn.Embedding(self.n_amino_acids, embedding_dim)
        self.hla_embedding = torch.nn.Embedding(self.n_amino_acids, embedding_dim)

        # BOS token
        self.bos_token = None
        self.bos_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=1)
        self.bos_transformer_encoder = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        
        # Peptide transformer and positional encoder
        self.pep_transformer_encoder = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.pep_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=peptide_length)
        self.pep_mask = None

        # HLA transformer and positional encoder
        self.hla_transformer_encoder = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers        
        )
        self.hla_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=allele_length)

        # PHLA transformer and positional encoder
        self.phla_transformer_encoder = nn.TransformerEncoder(
            encoder_layer= nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                batch_first=True
            ),
            num_layers=transformer_layers
        )
        self.phla_positional_encoder = self.PositionalEncoding(d_model=embedding_dim, max_len=self.seq_length)

        ## Projection model
        self.classifier = nn.Linear(embedding_dim, 1)

        ## Metrics and criterion
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task='binary')

    def network(self, x):
        # Separate peptide and HLA from input x
        x_hla = x[:,0:self.hparams.allele_length]
        x_pep = x[:,self.hparams.allele_length:]

        # Build BOS token
        '''
        if self.bos_token is None or self.bos_token.size(0) != len(x):
            bos_shape = (len(x), 1)
            token = torch.empty(bos_shape, dtype=torch.int32, device=x.device).fill_(0)
            self.bos_token = token
        '''
        '''
        # Build peptide mask
        if self.pep_mask is None or self.pep_mask.size(0) != len(x):
            mask_shape = (len(x), self.hparams.peptide_length)
            mask = torch.empty(mask_shape, device=x.device).fill_(0)
            mask = mask.masked_fill(x_pep == 22, float('-inf'))  # Update with blank token from aa_map
            self.src_mask = mask
        '''

        # BOS transformer
        '''
        x_bos = self.bos_embedding(self.bos_token)
        x_bos = self.bos_positional_encoder(x_bos)
        x_bos = self.bos_transformer_encoder(x_bos)
        '''

        # Peptide transformer network0
        x_pep = self.pep_embedding(x_pep)
        x_pep = self.pep_positional_encoder(x_pep)
        x_pep = self.pep_transformer_encoder(x_pep)

        # Peptide transformer network
        x_hla = self.hla_embedding(x_hla)
        x_hla = self.hla_positional_encoder(x_hla)
        x_hla = self.hla_transformer_encoder(x_hla)
        
        # Combine peptide and HLA representations
        x = torch.cat((x_pep, x_hla), dim=1)

        # Peptide transformer network
        x = self.phla_positional_encoder(x)
        x = self.phla_transformer_encoder(x)

        x = self.classifier(x[:,0,:])  # Classifier operates on BOS token only
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

        '''
        accuracy = self.accuracy(logits, labels.unsqueeze(-1))
        auroc = self.auroc(logits, labels.unsqueeze(-1))

        self.log("bce_loss", loss, on_epoch=True)
        self.log("accuracy", accuracy, on_epoch=True)
        self.log("auroc", auroc, on_epoch=True)
        '''

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.network(inputs)
        loss = self.criterion(logits, labels.unsqueeze(-1))
        
        accuracy = self.accuracy(logits, labels.unsqueeze(-1))
        auroc = self.auroc(logits, labels.unsqueeze(-1))

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)
        self.log("val_auroc", auroc, on_epoch=True)

        return loss
    
    # Predict needs work
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(torch.flatten(inputs, 1))
        embedding = self.embedding(inputs)
        return inputs, labels.unsqueeze(-1), outputs, embedding
