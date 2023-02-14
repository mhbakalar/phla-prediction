import os
from pathlib import Path

import lightning as L

from lightning.app.components import LightningTrainerMultiNode
from lightning.app.storage import Drive

from lightning.pytorch.callbacks import ModelCheckpoint

import models.datasets.phla_data as phla_data
import models.modules.transformer
import models.modules.split_transformer

import tbdrive

save_dir = "logs"
# Define parameters for sweep
embedding_dim = 128
transformer_heads = [32]
transformer_layers = [1]

# Configure data
hits_file = 'data/hits_16.txt'
decoys_file = 'data/decoys.txt'
aa_order_file = 'data/amino_acid_ordering.txt'
allele_sequence_file = 'data/alleles_95_variable.txt'

data = models.datasets.phla.PeptideHLADataModule(
    hits_file=hits_file,
    decoys_file=decoys_file,
    aa_order_file=aa_order_file,
    allele_sequence_file=allele_sequence_file,
    decoy_mul=1,
    train_test_split=0.2,
    batch_size=64,
    shuffle=True
)
data.prepare_data()

# Configure the model
heads = 1
layers = 1

model = models.modules.split_transformer.PeptideHLATransformer(
    peptide_length=12,
    allele_length=60,
    dropout_rate=0.3,
    embedding_dim=embedding_dim,
    transformer_heads=heads,
    transformer_layers=layers
)

# Run training
trainer = L.Trainer(
    max_epochs=10
)
trainer.fit(model, datamodule=data)
