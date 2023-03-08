import numpy as np
import pandas as pd

import lightning as L
import torch.utils.data as torch_data
import torch.nn.functional as F

from torch.utils.data import DataLoader

from .phla_numeric_data import NumericDataset
        
class DataModule(L.LightningDataModule):

    def __init__(
        self,
        hits_file: str,
        aa_order_file: str,
        allele_sequence_file: str,
        train_test_split: float,
        batch_size: int,
        predict_mode: bool=False,
        normalize: bool=False
    ):
        super().__init__()

        self.hits_file = hits_file
        self.aa_order_file = aa_order_file
        self.allele_sequence_file = allele_sequence_file
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.predict_mode = predict_mode
        self.normalize = normalize

    def prepare_data(self):
        # Set up peptide dataset
        self.peptide_dataset = NumericDataset(
            hits_file=self.hits_file,
            aa_order_file=self.aa_order_file,
            allele_sequence_file=self.allele_sequence_file,
            max_peptide_length=12, normalize=self.normalize)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        dataset_size = len(self.peptide_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor((1/self.num_splits) * dataset_size))

        # K-fold selection
        # choose fold to train on
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(indices)]
            
        if self.predict_mode:
            train_indices, val_indices = ([], indices)
        else:
            #train_indices, val_indices = all_splits[self.k]
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

        self.train_indices = train_indices
        self.val_indices = val_indices

    def train_dataloader(self):
        # Build subsampler and return data loader
        train_subsampler = torch_data.SubsetRandomSampler(self.train_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=train_subsampler, drop_last=True)

    def val_dataloader(self):
        # Build subsampler and return data loader
        val_subsampler = torch_data.SubsetRandomSampler(self.val_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def test_dataloader(self):
        # Build subsampler and return data loader
        val_subsampler = torch_data.SubsetRandomSampler(self.val_indices)
        return DataLoader(self.peptide_dataset, batch_size=self.batch_size, sampler=val_subsampler, drop_last=True)

    def predict_dataloader(self):
        val_subset = torch_data.Subset(self.peptide_dataset, self.val_indices)
        return DataLoader(val_subset, batch_size=self.batch_size)
