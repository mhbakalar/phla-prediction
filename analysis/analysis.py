#%% Import
import os
import glob
import torch
import pandas as pd
import tempfile
import numpy as np

from matplotlib import pyplot as plt
import umap

import lightning as L

from torch import nn


import models.datasets.phla
import models.modules.transformer

#%% Test Cell

def build_model(ckpt_path):
    # Create model to set parameters
    model = models.modules.transformer.Transformer(
        peptide_length=12,
        allele_length=60,
        dropout_rate=0.3,
        transformer_heads=4,
        transformer_layers=4
    )

    # Load the checkpoint and initialize model weights
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    return model

# Configure data
hits_file = 'data/hits_16_9.txt'
decoys_file = None
aa_order_file = 'data/amino_acid_ordering.txt'
allele_sequence_file = 'data/alleles_95_variable.txt'

data = models.datasets.phla.DataModule(
    hits_file=hits_file,
    decoys_file=decoys_file,
    aa_order_file=aa_order_file,
    allele_sequence_file=allele_sequence_file,
    decoy_mul=1,
    train_test_split=1.0,
    batch_size=64,
    shuffle=False
)
data.prepare_data()

# Build model from checkpoint
ckpt_path = glob.glob('logs/hits_95/lightning_logs/heads_4_layers_4/*.ckpt')[0]
model = build_model(ckpt_path)

# Monkey patch transformer model
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

attn_outputs = []
for module in model.transformer_encoder.modules():
    if isinstance(module, nn.MultiheadAttention):
        save_output = SaveOutput()
        patch_attention(module)
        module.register_forward_hook(save_output)
        attn_outputs.append(save_output)

# Run predictions
trainer = L.Trainer()
predictions = trainer.predict(model, datamodule=data)

layer = 1
batch = 1
attn_outputs[layer].outputs[batch].shape

# Analyze attention
layer = 1
batch = 15

for layer in [4]:
    batch_avg = attn_outputs[layer].outputs[batch].sum(axis=0)
    for head in [0,1,2,3]:
        attention = batch_avg[head][:,:]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(torch.transpose(attention, 0, 1), vmin=0, vmax=5)
        attn_sum = attention.sum(axis=0)

plt.plot(attn_sum)
#%% Test Cell
plt.plot(np.arange(0,10),np.arange(0,10))
plt.show()

#%%
# Variable site tomfoolery
test_data = attn_sum[0:60].numpy()
variable = np.loadtxt('data/variable_sites.txt', dtype='int32')
all_sites = np.arange(0,variable[-1]+1, dtype='int32')
df_sites = pd.DataFrame(index=all_sites)
df_sites.loc[:, 'values'] = 0
df_sites.loc[variable, 'values'] = test_data
df_sites.to_csv('output/attn_sum_head=1.txt', header=None, index=False)

# Alignment of HLA allele
df = pd.read_csv('data/alleles_95_variable.txt', header=None)
for allele in df[1]:
    print(allele)

# Combine batches and unravel prediction replicates
inputs, labels, outputs, embeddings = zip(*[batch for batch in predictions])

inputs = torch.vstack(list(inputs))
labels = torch.vstack(list(labels))
outputs = torch.vstack(list(outputs))
embeddings = torch.vstack(list(embeddings))

# Build prediction dataframe
peptides = [data.peptide_dataset.decode_peptide(p) for p in inputs]
pred_df = pd.DataFrame({'label':labels.flatten(), 'output':outputs.flatten(), 'hla_peptide':peptides})
pred_df['hla'] = pred_df['hla_peptide'].str.slice(0,60)
pred_df['peptide'] = pred_df['hla_peptide'].str.slice(60,60+12)
pred_df['length'] = pred_df['peptide'].str.len()

# Hack (update phla_datasets)
pred_df['length'] = pred_df['peptide'].str.find('x')


embed_vectors = []
for aa in data.peptide_dataset.aa_map:
    vector = embeddings[pred_df[pred_df['peptide'].str.slice(0,1) == aa].index][:,-12,:]
    if vector.shape[0] > 0:
        embed_vectors.append(vector[0,:])

embed_vectors = torch.vstack(embed_vectors)

reducer = umap.UMAP()
umap_map = reducer.fit_transform(embed_vectors.numpy())

fig, ax = plt.subplots()
x = umap_map[:,0]
y = umap_map[:,1]
n = list(data.peptide_dataset.aa_map.keys())[0:20]
ax.scatter(x, y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))

fig
