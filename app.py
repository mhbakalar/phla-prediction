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
import models.datasets.phla_binding
import models.modules.transformer

from models.modules.split_transformer import Transformer

import tbdrive

class PeptidePrediction(L.LightningWork):
    def __init__(self, *args, tb_drive, embedding_dim=128, heads=1, layers=1, save_dir="logs", load_from_checkpoint=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.drive = tb_drive
        self.cloud_build_config = L.BuildConfig()
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.layers = layers
        self.save_dir = save_dir
        self.load_from_checkpoint = load_from_checkpoint

        # Build configuration string
        self.config = "heads_{0}_layers_{1}".format(self.heads,self.layers)

    def _next_checkpoint_id(self):
        # Find latest version from Drive
        vid = 0
        version = self.config+"/version_{0}".format(vid)
        while self.save_dir+"/lightning_logs/"+version in self.drive.list(self.save_dir+"/lightning_logs/"+self.config):
            vid += 1
            version = self.config+"/version_{0}".format(vid)
        
        return vid
    
    def _get_latest_checkpoint(self, version):
        ckpt_options = self.drive.list(self.save_dir+"/lightning_logs/"+version)
        ckpt_options = fnmatch.filter(ckpt_options, '*.ckpt')
        ckpt_path = ckpt_options[-1]
        self.drive.get(ckpt_path, overwrite=True)   # Overwrite only needed for local execution
        return ckpt_path

    def run(self):
        # Find latest version from Drive
        vid = self._next_checkpoint_id()
        if(self.load_from_checkpoint):
            vid -= 1
        version = self.config+"/version_{0}".format(vid)
        print("Version: ", version)

        # Configure data
        hits_file = 'data/hits_95.txt'
        decoys_file = 'data/decoys.txt'
        aa_order_file = 'data/amino_acid_ordering.txt'
        allele_sequence_file = 'data/alleles_95_variable.txt'

        datamodule = models.datasets.phla_binding.DataModule(
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
        datamodule.prepare_data()

        # Configure the model
        model = None
        if(self.load_from_checkpoint):
            ckpt_path = ""
            try:
                ckpt_path = self._get_latest_checkpoint(version)
                print("Loading checkpoint path: ", ckpt_path)
                model = Transformer.load_from_checkpoint(ckpt_path)
            except:
                print("Could not load checkpoint from path: ", ckpt_path)
                print("Incrementing version.")
                vid += 1
                version = self.config+"/version_{0}".format(vid)
        
        if model == None:
            model = Transformer(
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
            save_dir=self.save_dir,
            version=version
        )

        checkpoint_callback = ModelCheckpoint(dirpath=logger.log_dir, save_top_k=2, monitor="val_loss")
        trainer = L.Trainer(
            max_epochs=5,
            logger=logger,
            callbacks=[checkpoint_callback],
            reload_dataloaders_every_n_epochs=1,
            accelerator="gpu"
        )
        trainer.fit(model, datamodule=datamodule)

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
    layers = config['layers'],
    load_from_checkpoint=True
)
app = L.LightningApp(component)