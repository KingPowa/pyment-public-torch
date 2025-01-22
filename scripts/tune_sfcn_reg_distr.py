# Final training fileimport os
import torch
import sys
import os
import hydra
import optuna

# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from omegaconf import OmegaConf
from pytorch_lightning.strategies import DDPStrategy

from pyment.utils.transforms import Resize3D
from pyment.utils.logging_tools import setup_logger, Session
from pyment.utils.optimize import Optimizer, instantiate_optim_config
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
                                       num_workers=config.slurm_config.num_workers,
                                       transforms=[Resize3D(IMG_SHAPE, "trilinear")]) 
    # Setup optimiziation
    logger.info(f"Setting up optimizer configurations...")

    optimizer_config = instantiate_optim_config(config)
    logger.info("Optimizer run description:")
    for part, part_config in optimizer_config.items():
        logger.info(f"-- SECTION: {part}")
        for param, descr in part_config.describe():
            logger.info(f"-- Param: {param} - {descr}")
            
    logger.info(f"Starting training and model arguments")
    logger.info(f"Number of epochs: {config.train_config.epochs}")
    logger.info(f"Setting up Wandbboard")
        
    # Model Arguments
    model_args = {
        "prediction_range": AGE_RANGE
    }
    trainer_args = {
        "accelerator" : session.accelerator,
        "devices" : torch.cuda.device_count(),  # Automatically detect available GPUs
        "num_nodes" : session.config.train_config.num_nodes,  # Number of nodes
        "max_epochs" : session.config.train_config.epochs,
        "strategy" : DDPStrategy(find_unused_parameters=False),
        "enable_progress_bar" : (not session.is_slurm())
    }
    trainable_args = {
        "pers_logger" : logger,
        "learning_rate" : config.trainable_config.lr,
        "momentum" : config.trainable_config.momentum,
        "milestones" : config.trainable_config.milestones,
        "weight_decay" : config.trainable_config.weight_decay
    }

    # Optimzer
    optimizer = Optimizer(
        session=session,
        trainer_args=trainer_args,
        model_args=model_args,
        trainable_args=trainable_args,
        model_class=RegressionSFCN,
        datamodule=dataloader,
        config=optimizer_config,
        logger=logger,
        monitor_metric=config.optuna_config.monitor_metric,
        callbacks=[]
    )

    logger.info("Starting optimizer...")
    logger.info(f"TRAINING INFO")
    logger.info(f"-- Devices {torch.cuda.device_count()}")
    logger.info(f"-- Slurm {session.is_slurm()}")
    logger.info(f"-- Accelerator {session.accelerator}")

    optimizer.optimize(
        n_trials = config.optuna_config.n_trials
    )

if __name__ == "__main__":
    # Check if the parameter is passed
    if len(sys.argv) != 2:
        print("Usage: python tune_sfcn_reg_distr.py")
        sys.exit(1)

    # Retrieve the parameter from the command line
    config_file = sys.argv[1]
    main(config_file = config_file)
