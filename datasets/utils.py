from omegaconf import ListConfig, DictConfig
from logging import Logger
from datasets.file_based import MedicalDataset
import hydra

def instantiate_datasets(datasets: ListConfig[DictConfig], logger: Logger):
    inst_datasets = []
    logger.info("Selected datasets:")
    for dataset in datasets:
        inst_dataset: MedicalDataset = hydra.utils.instantiate(dataset)
        inst_datasets.append(inst_dataset)
        logger.info(f"-- {inst_dataset.get_name()} (Location: {inst_dataset.get_location()})")
    return inst_datasets