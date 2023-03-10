import numpy as np
import pandas as pd

import lightning as L
import torch
import torch.utils.data as torch_data
import torch.nn.functional as F

from torch.utils.data import Dataset

class BindingDataset(Dataset):

    def __init__(self, hits_file, decoys_file, aa_order_file,
        allele_sequence_file, decoy_mul, max_peptide_length=12,
        cterm_marker='.'
    ):
        ## AA order for encoding defined in file
        aa_ordering = pd.read_csv(aa_order_file, header=None)
        self.inverse_aa_map = aa_ordering[0].to_dict()
        self.aa_map = {v: k for k, v in self.inverse_aa_map.items()}

        ## Read allele sequence from file
        allele_data = pd.read_csv(allele_sequence_file, header=None, names=['allele','seq'])

        # Max length of peptide + HLA in the dataset
        allele_length = allele_data['seq'].str.len()[0]
        self.max_length = allele_length + max_peptide_length

        ## Read hits and decoys from file
        hits = pd.read_csv(hits_file, sep=' ')
        hit_peptides = hits['seq'] + cterm_marker

        ## Read allele sequences
        hits_allele_data = allele_data.set_index('allele').loc[hits['allele']]
        self.hits_hla_peptides = hits_allele_data['seq'].values + hit_peptides.values

        # Generate decoys
        if decoys_file is not None:
            decoys = pd.read_csv(decoys_file, sep=' ')
            decoy_peptides = decoys['seq'] + cterm_marker

            decoy_peptides = np.random.choice(decoy_peptides, hits.shape[0]*decoy_mul, replace=False)
            decoy_alleles = np.random.choice(hits_allele_data['seq'], hits.shape[0]*decoy_mul, replace=True)
            self.decs_hla_peptides = decoy_alleles + decoy_peptides

            ## Assemble dataset composed of hits and decoys
            self.peptides = np.hstack([self.hits_hla_peptides, self.decs_hla_peptides])
            self.binds = np.hstack([torch.ones(len(self.hits_hla_peptides)), torch.zeros(len(self.decs_hla_peptides))])
        else:
            ## Assemble dataset composed of hits only
            self.peptides = self.hits_hla_peptides
            self.binds = torch.ones(len(self.hits_hla_peptides))

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
        return self.encoded_peptides[idx], self.binds[idx]

    def decode_peptide(self, encoded):
        try:
            peptide = ''.join([self.inverse_aa_map[aa.item()] for aa in encoded])
            return peptide
        except:
            return ''