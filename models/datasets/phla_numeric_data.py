import numpy as np
import pandas as pd

import lightning as L
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

class NumericDataset(Dataset):

    def __init__(
        self, 
        hits_file: str, 
        aa_order_file: str,
        allele_sequence_file: str, 
        max_peptide_length: int=12,
        cterm_marker: str='.', 
        normalize=False
    ):
        ## AA order for encoding defined in file
        aa_ordering = pd.read_csv(aa_order_file, header=None)
        self.inverse_aa_map = aa_ordering[0].to_dict()
        self.aa_map = {v: k for k, v in self.inverse_aa_map.items()}

        ## Read allele sequence from file
        allele_data = pd.read_csv(allele_sequence_file, header=None, names=['allele','seq'])

        ## Read numeric data from file

        ## Max length of peptide + HLA in the dataset
        allele_length = allele_data['seq'].str.len()[0]
        self.max_length = allele_length + max_peptide_length

        ## Read hits and decoys from file
        hits = pd.read_csv(hits_file, sep=' ')
        hit_peptides = hits['seq'] + cterm_marker

        ## Map allele sequences and append to peptides
        hits_allele_data = allele_data.set_index('allele').loc[hits['allele']]
        self.hits_hla_peptides = hits_allele_data['seq'].values + hit_peptides.values

        ## Assemble dataset composed of hits only
        self.peptides = self.hits_hla_peptides
        self.values = torch.tensor(hits['val'], dtype=torch.float32)

        if normalize:
            mean, std, var = torch.mean(self.values), torch.std(self.values), torch.var(self.values)
            self.values = (self.values-mean)/std

        ## Precompute encoding for peptides
        self._encode_peptides(self.peptides)

    def _encode_peptides(self, peptides):
        self.encoded_peptides = []
        cterm_index = len(self.aa_map.keys())
        for peptide in peptides:
            aa_encoding = torch.as_tensor([self.aa_map[aa] for aa in peptide])
            aa_encoding = F.pad(aa_encoding, pad=(0,self.max_length-aa_encoding.shape[0]), value=cterm_index)  # Update with len(aa_map)
            self.encoded_peptides.append(aa_encoding)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        return self.encoded_peptides[idx], self.values[idx]

    def decode_peptide(self, encoded):
        try:
            peptide = ''.join([self.inverse_aa_map[aa.item()] for aa in encoded])
            return peptide
        except:
            return ''