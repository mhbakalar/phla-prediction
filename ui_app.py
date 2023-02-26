import os
import glob
import torch
import pandas as pd
import tempfile

from matplotlib import pyplot as plt
import umap

import lightning as L
from lightning.app.components import ServeGradio
from lightning.app.storage import Drive

import models.datasets.phla_binding as phla_binding
import models.phla_transformer as phla_transformer

import gradio as gr

class LitGradio(ServeGradio):

    inputs = [gr.inputs.Textbox(default='A0201', label='Allele'), gr.inputs.Textbox(default='', label='Peptides', lines=10)]
    outputs = [gr.outputs.Textbox(label='output'), gr.Plot()]
    examples = [['A0201', 'GILGFVFTL']]

    def __init__(self, drive, ckpt_path):
        super().__init__()

        self.drive = drive
        self.ckpt_path = ckpt_path

    def prepare_hits(self):
        # Configure data
        hits_file = 'data/hits_16.txt'
        decoys_file = None
        aa_order_file = 'data/amino_acid_ordering.txt'
        allele_sequence_file = 'data/alleles_95_variable.txt'

        data = phla_binding.DataModule(
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

        # Run predictions
        trainer = L.Trainer()
        predictions = trainer.predict(self.model, datamodule=data)
        inputs, labels, outputs, embeddings = zip(*[batch for batch in predictions])

        # Analyze data
        embeddings = torch.vstack([e.flatten(start_dim=1) for e in embeddings])
        print("Embedding shape:", embeddings.shape)

        reducer = umap.UMAP()
        umap_map = reducer.fit_transform(embeddings.numpy())
        print(umap_map.shape)

        #outputs = torch.vstack(list(outputs))

    def plot_peptides(self, embeddings):
        self.prepare_hits()
        print(embeddings[0].flatten().shape)
        df_pens = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')
        fig = plt.figure()
        plt.scatter(x=df_pens['bill_length_mm'], y=df_pens['bill_depth_mm'])
        return fig

    def predict(self, allele_in, peptides_in):
        peptides = peptides_in.splitlines()
        allele_data = pd.read_csv('data/alleles_95_variable.txt', header=None, names=['allele','seq'])
        allele_seq = allele_data[allele_data['allele'] == allele_in]['seq']

        hla_peptide = [allele_seq+p+'x' for p in peptides]
        peptide_lengths = [len(p) for p in peptides]

        tmp_hits = tempfile.NamedTemporaryFile()

        # Open the file for writing.
        with open(tmp_hits.name, 'w') as f:
            f.write('allele len seq\n')
            for i, pep in enumerate(peptides):
                f.write('{0} {1} {2}\n'.format(allele_in, peptide_lengths[i], pep))

        # Configure data
        hits_file = tmp_hits.name
        decoys_file = None
        aa_order_file = 'data/amino_acid_ordering.txt'
        allele_sequence_file = 'data/alleles_95_variable.txt'

        data = phla_binding.DataModule(
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

        # Run predictions
        trainer = L.Trainer()
        predictions = trainer.predict(self.model, datamodule=data)

        # Analyze data
        inputs, labels, outputs, embeddings = zip(*[batch for batch in predictions])
        outputs = torch.vstack(list(outputs))

        output_data = pd.DataFrame({'allele':[allele_in]*len(peptides), 'seq':peptides, 'pred':outputs.flatten().tolist()})

        # Create plot
        data_plot = self.plot_peptides(embeddings)
        print(embeddings[0].flatten(start_dim=1))

        return output_data.to_string(), data_plot

    def build_model(self):
        # Create model to set parameters
        model = phla_transformer.Transformer(
            peptide_length=12,
            allele_length=60,
            dropout_rate=0.3,
            transformer_heads=4,
            transformer_layers=8
        )

        # Load the checkpoint and initialize model weights
        #drive.get(self.ckpt_path)
        checkpoint = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        return model

class RootFlow(L.LightningFlow):
    def __init__(self, drive, ckpt_path):
        super().__init__()
        self.lit_gradio = LitGradio(drive, ckpt_path)

    def run(self):
        self.lit_gradio.run()

    def configure_layout(self):
        return [{"name": "home", "content": self.lit_gradio}]

drive = Drive("lit://hits_95", component_name="pmhc")
ckpt_path = glob.glob('logs/hits_95/lightning_logs/heads_8_layers_3/*.ckpt')[0]
app = L.LightningApp(RootFlow(drive, ckpt_path))
