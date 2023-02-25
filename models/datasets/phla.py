import numpy as np
import pandas as pd

import lightning as L
import torch.utils.data as torch_data
import torch.nn.functional as F

from torch.utils.data import DataLoader

import phla_data
        
class DataModule(L.LightningDataModule):

    def __init__(
        self,
        hits_file: str,
        decoys_file: str,
        aa_order_file: str,
        allele_sequence_file: str,
        decoy_mul: int,
        decoy_pool_mul: int,
        train_test_split: float,
        batch_size: int,
        predict_mode: bool=False,
    ):
        super().__init__()

        self.hits_file = hits_file
        self.decoys_file = decoys_file
        self.aa_order_file = aa_order_file
        self.allele_sequence_file = allele_sequence_file
        self.decoy_mul = decoy_mul
        self.decoy_pool_mul = decoy_pool_mul
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.predict_mode = predict_mode

    def prepare_data(self):
        # Set up peptide dataset
        self.peptide_dataset = phla_data.PeptideHLADataset(
            hits_file=self.hits_file,
            decoys_file=self.decoys_file,
            aa_order_file=self.aa_order_file,
            allele_sequence_file=self.allele_sequence_file,
            decoy_mul=self.decoy_pool_mul,
            max_peptide_length=12)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        dataset_size = len(self.peptide_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.train_test_split * dataset_size))
            
        if self.predict_mode:
            train_indices, val_indices = ([], indices)
        else:
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

        self.train_indices = train_indices
        self.val_indices = val_indices

    """
    Only returns positive values for validation set. MHB...
    """
    def rand_balance_data(self, indices):
        # Extracts the binding labels for a subset of the training data
        labels = self.peptide_dataset.binds[indices]

        # Create a list of indices for the training samples that have a positive/negative binding label
        binder_indices = labels.nonzero()[0]
        decoy_indices = (labels == 0).nonzero()[0]

        # Selects a subset of negative training samples to balance samples for training
        subset_size = min(len(binder_indices)*self.decoy_mul, len(decoy_indices))
        decoy_subset = np.random.choice(decoy_indices, subset_size, replace=False)

        # Contatenate binders and decoys and select indices in original order
        balanced_indices = np.hstack((binder_indices, decoy_subset))
        balanced_indices = np.array(indices)[np.in1d(indices, balanced_indices)]

        return balanced_indices

    def train_dataloader(self):
        # Balance decoy data
        #balanced_indices = self.rand_balance_data(self.train_indices)

        # Build subsampler and return data loader
        train_subsampler = torch_data.SubsetRandomSampler(self.train_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=train_subsampler, drop_last=True)

    def val_dataloader(self):
        # Balance decoy data
        #balanced_indices = self.rand_balance_data(self.val_indices)

        # Build subsampler and return data loader
        val_subsampler = torch_data.SubsetRandomSampler(self.val_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def test_dataloader(self):
        # Balance decoy data
        balanced_indices = self.rand_balance_data(self.val_indices)

        # Build subsampler and return data loader
        val_subsampler = torch_data.SubsetRandomSampler(balanced_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def predict_dataloader(self):
        val_subset = torch_data.Subset(self.peptide_dataset, self.val_indices)
        return DataLoader(val_subset, batch_size=self.batch_size)
