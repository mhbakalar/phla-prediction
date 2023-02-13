import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

class PeptideDataset(Dataset):

    def __init__(self, hits_file, decoys_file, aa_order_file, decoy_mul, allele, max_peptide_length, padding_key):

        ## AA order for encoding defined in file
        aa_ordering = pd.read_csv(aa_order_file, header=None)
        self.inverse_aa_map = aa_ordering[0].to_dict()
        self.aa_map = {v: k for k, v in self.inverse_aa_map.items()}

        # Add a character for C terminus
        self.aa_map['x'] = 20
        self.inverse_aa_map[20] = 'x'

        # Max length of peptides in the dataset
        self.padding_key = padding_key
        self.max_peptide_length = max_peptide_length

        ## Read hits and decoys from file
        hits = pd.read_csv(hits_file, sep=' ')
        hits = hits[(hits['allele'] == allele)]
        self.hits_peptides = hits['seq'].values
        decoys = pd.read_csv(decoys_file, sep=' ')['seq'].values
        self.decs_peptides = np.random.choice(decoys, self.hits_peptides.size*decoy_mul, replace=False)

        self.hits_peptides = self.hits_peptides + 'x'
        self.decs_peptides = self.decs_peptides + 'x'

        ## Assemble dataset composed of hits and decoys
        self.peptides = np.hstack([self.hits_peptides, self.decs_peptides])
        self.binds = np.hstack([torch.ones(len(self.hits_peptides)), torch.zeros(len(self.decs_peptides))])

        ## Precompute encoding for peptides
        self._encode_peptides(self.peptides)

    def _encode_peptides(self, peptides):
        self.encoded_peptides = []
        for peptide in peptides:
            aa_encoding = torch.as_tensor([self.aa_map[aa] for aa in peptide])
            aa_encoding = F.pad(aa_encoding, pad=(0,self.max_peptide_length-aa_encoding.shape[0]), value=21)
            self.encoded_peptides.append(aa_encoding)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        return self.encoded_peptides[idx], self.binds[idx]

    def decode_peptide(self, encoded):
        peptide = ''.join([self.inverse_aa_map[aa.item()] for aa in encoded])
        return peptide
