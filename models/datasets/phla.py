import numpy as np
import pandas as pd

import lightning as L
import torch
import torch.utils.data as torch_data
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
        
class PeptideHLADataModule(L.LightningDataModule):

    class PeptideHLADataset(Dataset):

        def __init__(self, hits_file, decoys_file, aa_order_file,
            allele_sequence_file, decoy_mul, max_peptide_length=12,
            use_core=False, cterm_marker='.'
        ):
            ## AA order for encoding defined in file
            aa_ordering = pd.read_csv(aa_order_file, header=None)
            self.inverse_aa_map = aa_ordering[0].to_dict()
            self.aa_map = {v: k for k, v in self.inverse_aa_map.items()}

            ## Read allele sequence from file
            allele_data = pd.read_csv(allele_sequence_file, header=None, names=['allele','seq'])

            # Max length of peptides in the dataset
            allele_length = allele_data['seq'].str.len()[0]
            if use_core:
                self.max_length = allele_length + 8
            else:
                self.max_length = allele_length + max_peptide_length

            ## Helper function for extracting peptide core 8-MER
            pep_core_fun = lambda x: x.str.slice(0,4) + x.str.slice(start=-4)

            ## Read hits and decoys from file
            hits = pd.read_csv(hits_file, sep=' ')
            if use_core:
                hit_peptides = pep_core_fun(hits['seq'])
            else:
                hit_peptides = hits['seq'] + cterm_marker

            ## Read allele sequences
            hits_allele_data = allele_data.set_index('allele').loc[hits['allele']]
            self.hits_hla_peptides = hits_allele_data['seq'].values + hit_peptides.values

            # Generate decoys
            if decoys_file is not None:
                decoys = pd.read_csv(decoys_file, sep=' ')
                if use_core:
                    decoy_peptides = pep_core_fun(decoys['seq'])
                else:
                    decoy_peptides = decoys['seq'] + cterm_marker

                decoy_peptides = np.random.choice(decoy_peptides, hits.shape[0]*decoy_mul, replace=False)
                decoy_alleles = np.random.choice(hits_allele_data['seq'], hits.shape[0]*decoy_mul, replace=False)
                self.decs_hla_peptides = decoy_alleles + decoy_peptides

                ## Assemble dataset composed of hits and decoys
                self.peptides = np.hstack([self.hits_hla_peptides, self.decs_hla_peptides])
                self.binds = np.hstack([torch.ones(len(self.hits_hla_peptides)), torch.zeros(len(self.decs_hla_peptides))])
            else:
                ## Assemble dataset composed of hits and decoys
                self.peptides = self.hits_hla_peptides
                self.binds = torch.ones(len(self.hits_hla_peptides))

            ## Precompute encoding for peptides
            self._encode_peptides(self.peptides)

        def _encode_peptides(self, peptides):
            self.encoded_peptides = []
            for peptide in peptides:
                aa_encoding = torch.as_tensor([self.aa_map[aa] for aa in peptide])
                aa_encoding = F.pad(aa_encoding, pad=(0,self.max_length-aa_encoding.shape[0]), value=21)
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

    def __init__(
        self,
        hits_file: str,
        decoys_file: str,
        aa_order_file: str,
        allele_sequence_file: str,
        decoy_mul: int,
        train_test_split: float,
        batch_size: int,
        use_core: bool=False,
        shuffle: bool=False,
    ):
        super().__init__()
        self.hits_file = hits_file
        self.decoys_file = decoys_file
        self.aa_order_file = aa_order_file
        self.allele_sequence_file = allele_sequence_file
        self.decoy_mul = decoy_mul
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.use_core = use_core
        self.shuffle = shuffle

    def prepare_data(self):
        # Set up peptide dataset
        self.peptide_dataset = self.PeptideHLADataset(
            hits_file=self.hits_file,
            decoys_file=self.decoys_file,
            aa_order_file=self.aa_order_file,
            allele_sequence_file=self.allele_sequence_file,
            decoy_mul=self.decoy_mul,
            max_peptide_length=12,
            use_core=self.use_core)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        dataset_size = len(self.peptide_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.train_test_split * dataset_size))
        if self.shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        self.train_indices = train_indices
        self.val_indices = val_indices

    def train_dataloader(self):
        train_subsampler = torch_data.SubsetRandomSampler(self.train_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=train_subsampler, drop_last=True)

    def val_dataloader(self):
        val_subsampler = torch_data.SubsetRandomSampler(self.val_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def test_dataloader(self):
        val_subsampler = torch_data.SubsetRandomSampler(self.val_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def predict_dataloader(self):
        val_subset = torch_data.Subset(self.peptide_dataset, self.val_indices)
        return DataLoader(val_subset, batch_size=self.batch_size)
