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
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping

from pyment.utils.transforms import Resize3D
from pyment.utils.sbatch_utils import query_lr
from pyment.utils.logging_tools import setup_logger, Session, AdvancedModelCheckpoint, AdvancedWandLogger
from pyment.datasets.utils import instantiate_datasets
from pyment.dataloaders.mri import MRIHoldoutDataLoader

from pyment.model.trainable_sfcn import SFCNModule
from pyment.model.sfcn_reg import RegressionSFCN

def main(config_file: str):
    # Constants
    CONFIG_PATH = config_file
    # Load configuration
    config = OmegaConf.load(CONFIG_PATH)
    # Setup Session
    session = Session("SFCN_training", config=config)
    # Setup Logger
    logger = setup_logger(session)
    # Setup training vars
    IMG_SHAPE = config.train_config.img_shape
    AGE_RANGE = config.train_config.age_range
    # Setup datasets
    datasets = instantiate_datasets(config.datasets, logger)
    dataloader = MRIHoldoutDataLoader(dataset=datasets, 
                                       batch_size=config.train_config.batch_size,
                                       num_workers=config.train_config.num_workers,
                                       transforms=[Resize3D(IMG_SHAPE, "trilinear")]) 
    # Setup Model
    logger.info(f"Declaring SFCN model...")

    # Querying LR from env
    lr = query_lr()
    if lr is not None:
        logger.info(f"Learning rate passed from env: {lr}")
    else:
        logger.info(f"Using default learning rate passed from config: {config.train_config.lr}")
        lr = config.train_config.lr

    model = SFCNModule(RegressionSFCN(prediction_range=AGE_RANGE), learning_rate=lr, pers_logger=logger,
                       milestones=config.train_config.milestones, decay=config.train_config.decay,
                       max_epochs=session.config.train_config.epochs)
    # Setup Training
    logger.info(f"Starting training. Number of epochs: {config.train_config.epochs}")
    logger.info(f"Setting up Wandbboard")
    wand_logger = AdvancedWandLogger(model, session)
    checkpoint_callback = AdvancedModelCheckpoint(session=session,
                                            filename_suffix='holdout',
                                            monitor='val_loss',
                                            mode='min')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.train_config.patience,           # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'
    )
    logger.info(f"Setting up Trainer")
    logger.info(f"TRAINING INFO")
    logger.info(f"-- Devices {torch.cuda.device_count()}")
    logger.info(f"-- Slurm {session.is_slurm()}")
    logger.info(f"-- Accelerator {session.accelerator}")
    trainer = Trainer(
        accelerator=session.accelerator,
        devices=torch.cuda.device_count(),  # Automatically detect available GPUs
        num_nodes=session.config.train_config.num_nodes,  # Number of nodes
        max_epochs=session.config.train_config.epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wand_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
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
