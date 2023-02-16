import os
from pathlib import Path
from pickle import dump

import lightning as L
import torch

from lightning.app.components import LightningTrainerMultiNode
from lightning.app.storage import Drive

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
        embedding_dim = 256
        transformer_heads = [16]
        transformer_layers = [3]

        # Parameter sweepq
        for heads in transformer_heads:
            for layers in transformer_layers:
                # Find latest version from Drive
                vid = 0
                config = "heads_{0}_layers_{1}".format(heads,layers)
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
                    batch_size=64,
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
                    transformer_layers=layers
                )

                # Create a logger
                logger = tbdrive.DriveTensorBoardLogger(
                    drive=self.drive,
                    save_dir=save_dir,
                    version=version
                )

                # Setup trainer and run training
                class MemoryTracker(Callback):
                    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                        snapshot = torch.cuda.memory_snapshot()
                        with open(trainer.log_dir+'/snapshot.pickle', 'wb') as f:
                            dump(snapshot, f)

                checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val_auroc")
                # memory_callback = MemoryTracker()

                trainer = L.Trainer(
                    max_epochs=20,
                    logger=logger,
                    callbacks=[checkpoint_callback],
                    accelerator="gpu",
                    reload_dataloaders_every_n_epochs=1,
                    auto_scale_batch_size="power"
                )
                trainer.fit(model, datamodule=data)

                # Manually upload the log state to Drive
                try:
                    logger._upload_to_storage(logs_only=False)
                except Exception as e:
                    print(e)


drive = Drive("lit://hits_95", component_name="pmhc")

component = LightningTrainerMultiNode(
    PeptidePrediction,
    num_nodes=1,
    cloud_compute=L.CloudCompute(
        "gpu",
        idle_timeout=5
    ),
    tb_drive=drive
)
app = L.LightningApp(component)
