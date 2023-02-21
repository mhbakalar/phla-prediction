from sklearn.model_selection import ParameterGrid

import lightning as L

from lightning.app.components import LightningTrainerMultiNode
from lightning.app.storage import Drive

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback

import models.datasets.phla_data
import models.modules.transformer
import models.modules.split_transformer

import tbdrive

class PeptidePrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, embedding_dim=128, heads=1, layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = tb_drive
        self.cloud_build_config = L.BuildConfig()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.layers = layers

    def run(self):
        save_dir = "logs"

        # Find latest version from Drive
        vid = 0
        config = "heads_{0}_layers_{1}".format(self.heads,self.layers)
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
            decoy_pool_mul=1,
            train_test_split=0.2,
            batch_size=64,
            predict_mode=False
        )
        data.prepare_data()

        # Configure the model
        model = models.modules.split_transformer.PeptideHLATransformer(
            peptide_length=12,
            allele_length=34,
            dropout_rate=0.3,
            embedding_dim=self.embedding_dim,
            transformer_heads=self.heads,
            transformer_layers=self.layers,
            learning_rate=1e-4
        )

        # Create a logger
        logger = tbdrive.DriveTensorBoardLogger(
            drive=self.drive,
            save_dir=save_dir,
            version=version
        )

        checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val_loss")
        trainer = L.Trainer(
            max_epochs=15,
            logger=logger,
            callbacks=[checkpoint_callback],
            reload_dataloaders_every_n_epochs=1,
            accelerator="gpu"
        )
        trainer.fit(model, datamodule=data)

        # Manually upload the log state to Drive
        try:
            print("Final upload including checkpoint files...")
            logger._upload_to_storage(logs_only=False)
        except Exception as e:
            print(e)

config = {'embedding_dim': 200, 'heads': 25, 'layers': 1}

component = LightningTrainerMultiNode(
    PeptidePrediction,
    num_nodes=1,
    cloud_compute=L.CloudCompute("gpu"),
    tb_drive=Drive("lit://hits_95", component_name="pmhc"),
    embedding_dim=config['embedding_dim'],
    heads=config['heads'],
    layers = config['layers']
)
app = L.LightningApp(component)