# Final training fileimport os
import torch
import sys
import os
import hydra

# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from configuration.files import LMDBDatasetConfig
from utils.transforms import Cropper, Resize3D
from utils.logging_tools import setup_logger, Session, AdvancedModelCheckpoint, AdvancedWandLogger
from datasets.file_based import LMDBDataset
from datasets.mri import MRIDataset
from datasets.utils import instantiate_datasets
from dataloaders.mri import MRIHoldoutDataLoader

from model.trainable_sfcn import SFCNModule
from model.sfcn_reg import RegressionSFCN

def main(config_file: str):
    # Constants
    CONFIG_PATH = config_file
    # Load configuration
    config = OmegaConf.load(CONFIG_PATH)
    # Setup Session
    session = Session("SFCN_training", config=config)
    # Setup directory
    working_directory = session.working_directory
    # Setup Logger
    logger = setup_logger(session)
    # Setup training vars
    IMG_SHAPE = config.train_config.img_shape
    AGE_RANGE = config.train_config.age_range
    # Setup datasets
    datasets = instantiate_datasets(config.datasets, logger)
    dataloader = MRIHoldoutDataLoader(dataset=datasets, 
                                       batch_size=16,
                                       num_workers=0,
                                       transforms=[Resize3D(IMG_SHAPE, "trilinear")]) 
    # Setup Model
    logger.info(f"Declaring SFCN model...")
    model = RegressionSFCN(prediction_range=AGE_RANGE)
    # Setup Training
    logger.info(f"Starting training. Number of epochs: {config.train_config.epochs}")
    logger.info(f"Setting up Wandbboard")
    wand_logger = AdvancedWandLogger(model, session)
    checkpoint_callback = AdvancedModelCheckpoint(session=session,
                                            filename_suffix='holdout',
                                            monitor='val_loss',
                                            mode='min')
    logger.info(f"Setting up Trainer")
    trainer = Trainer(
        max_epochs=session.config.train_config.epochs,
        devices=session.devices,
        accelerator=session.accelerator,
        logger=wand_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=(not session.is_slurm())
    )

    logger.info(f"Starting trainer")
    trainer.fit(model, datamodule=dataloader)

    performances = trainer.callback_metrics
    logger.info(f"Finished. Total performance: {performances}")

if __name__ == "__main__":
    # Check if the parameter is passed
    if len(sys.argv) != 2:
        print("Usage: python train_sfcn_reg_distr.py")
        sys.exit(1)

    # Retrieve the parameter from the command line
    config_file = sys.argv[1]
    main(config_file = config_file)
