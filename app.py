import os
import sys
import psutil
from pathlib import Path
from pickle import dump
from sklearn.model_selection import ParameterGrid

import lightning as L
import torch

from lightning.app.components import LightningTrainerMultiNode
from lightning.app.storage import Drive

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import models.datasets.phla_data
import models.modules.transformer
import models.modules.split_transformer


import tbdrive

class PeptidePrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = tb_drive
        self.cloud_build_config = L.BuildConfig()

    def run(self):
        save_dir = "logs"

        # Define parameters for sweep
        parameter_dict = {'embedding_dim': [256],
                          'heads': [16],
                          'l0_layers': [1,3],
                          'l1_layers': [3]
                          }

        # Parameter sweep
        for params in ParameterGrid(parameter_dict):
            # Extract parameters
            embedding_dim = params['embedding_dim']
            heads = params['heads']
            l0_layers = params['l0_layers']
            l1_layers = params['l1_layers']

            # Find latest version from Drive
            vid = 0
            config = "heads_{0}_l0_{1}_l1_{2}".format(heads,l0_layers,l1_layers)
            version = config+"/version_{0}".format(vid)
            while save_dir+"/lightning_logs/"+version in self.drive.list(save_dir+"/lightning_logs/"+config):
                vid += 1
                version = config+"/version_{0}".format(vid)

            # Configure data
            hits_file = 'data/hits_95.txt'
            decoys_file = 'data/decoys.txt'
            aa_order_file = 'data/amino_acid_ordering.txt'
            allele_sequence_file = 'data/alleles_95_variable.txt'

            data = models.datasets.phla_data.PeptideHLADataModule(
                hits_file=hits_file,
                decoys_file=decoys_file,
                aa_order_file=aa_order_file,
                allele_sequence_file=allele_sequence_file,
                decoy_mul=1,
                decoy_pool_mul=10,
                train_test_split=0.2,
                batch_size=32,
                predict_mode=False
            )
            data.prepare_data()

            # Configure the model
            model = models.modules.split_transformer.PeptideHLATransformer(
                peptide_length=12,
                allele_length=60,
                dropout_rate=0.3,
                embedding_dim=embedding_dim,
                transformer_heads=heads,
                transformer_l0_layers=l0_layers,
                transformer_l1_layers=l1_layers
            )

            # Create a logger
            logger = tbdrive.DriveTensorBoardLogger(
                drive=self.drive,
                save_dir=save_dir,
                version=version
            )

            class MemoryCallback(Callback):
                
                def on_train_epoch_end(self, trainer, pl_module):
                    def cpuStats():
                        print(sys.version)
                        print(psutil.cpu_percent())
                        print(psutil.virtual_memory())  # physical memory usage
                        pid = os.getpid()
                        py = psutil.Process(pid)
                        memory_use = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
                        print('memory GB:', memory_use)

                    max_allocated = torch.cuda.max_memory_allocated()
                    if max_allocated != 0:
                        print('\nMemory usage: ', torch.cuda.memory_allocated() / max_allocated)
                    cpuStats()

                    

            memory_callback = MemoryCallback()
            checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val_loss")
            trainer = L.Trainer(
                max_epochs=10,
                logger=logger,
                callbacks=[checkpoint_callback, memory_callback],
                accelerator="gpu",
                reload_dataloaders_every_n_epochs=1
            )
            #trainer.tune(model, datamodule=data)
            trainer.fit(model, datamodule=data)

            # Manually upload the log state to Drive
            try:
                print("Final upload including checkpoint files...")
                logger._upload_to_storage(logs_only=False)
            except Exception as e:
                print(e)


drive = Drive("lit://hits_95", component_name="pmhc")

component = LightningTrainerMultiNode(
    PeptidePrediction,
    num_nodes=1,
    cloud_compute=L.CloudCompute(
        "gpu"
    ),
    tb_drive=drive
)
app = L.LightningApp(component)
