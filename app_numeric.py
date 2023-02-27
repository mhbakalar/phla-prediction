import sys
sys.path.append("/content")
import fnmatch

from sklearn.model_selection import ParameterGrid

import lightning as L

from lightning.app.components import LightningTrainerMultiNode
from lightning.app.storage import Drive

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import models
import models.datasets
import models.modules.split_transformer
import models.modules.numeric_transformer

import models.datasets.phla_numeric

import tbdrive

class PeptidePrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, save_dir="logs", **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = tb_drive
        self.cloud_build_config = L.BuildConfig()
        self.save_dir = save_dir
        self.load_from_checkpoint = True

    def run(self):
        # Configure data
        hits_file = 'data/tstab_data.txt'
        aa_order_file = 'data/amino_acid_ordering.txt'
        allele_sequence_file = 'data/alleles_95_variable.txt'

        data = models.datasets.phla_numeric.DataModule(
            hits_file=hits_file,
            aa_order_file=aa_order_file,
            allele_sequence_file=allele_sequence_file,
            train_test_split=0.2,
            batch_size=64,
            predict_mode=False
        )
        data.prepare_data()

        # Configure the model
        ckpt_path = "logs/lightning_logs/heads_8_layers_3/version_0/epoch=13-step=59920.ckpt"
        version = "heads_8_layers_3/numeric"
        model = models.modules.split_transformer.Transformer.load_from_checkpoint(ckpt_path)
        model.embedding_dim = 200
        
        # Build the transfer learning model
        transfer_model = models.modules.numeric_transformer.Transformer(model, embedding_dim=model.embedding_dim)

        # Create a logger
        logger = tbdrive.DriveTensorBoardLogger(
            drive=self.drive,
            save_dir=self.save_dir,
            version=version
        )

        checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val_loss")
        trainer = L.Trainer(
            max_epochs=15,
            logger=logger,
            reload_dataloaders_every_n_epochs=1,
            callbacks=[checkpoint_callback],
            accelerator="cpu"
        )
        trainer.fit(transfer_model, datamodule=data)

        # Manually upload the log state to Drive
        try:
            print("Final upload including checkpoint files...")
            logger._upload_to_storage(logs_only=False)
        except Exception as e:
            print(e)


component = LightningTrainerMultiNode(
    PeptidePrediction,
    num_nodes=1,
    cloud_compute=L.CloudCompute("gpu"),
    tb_drive=Drive("lit://hits_95", component_name="pmhc")
)
app = L.LightningApp(component)