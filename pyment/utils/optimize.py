import optuna
from optuna.integration import PyTorchLightningPruningCallback

from typing import Type, Any, Callable
from copy import deepcopy
from abc import abstractmethod
from omegaconf import DictConfig
import itertools
from hydra.utils import get_method
from hydra.utils import instantiate

from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback

from .logging_tools import RankZeroLogger
from ..model.sfcn import SFCN
from ..model.trainable_sfcn import SFCNModule

class Param:

    def __init__(self, name: str, structure: list):
        self.name = name
        self.read_structure(structure)

    @abstractmethod
    def read_structure(self, structure: list):
        raise NotImplementedError

    @abstractmethod
    def suggest(self, trial: optuna.trial.Trial) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
    
class IntParam(Param):

    def read_structure(self, structure: list):
        num_args = len(structure)
        assert num_args >= 2
        self.low = structure[0]
        self.high = structure[1]
        if num_args >= 3:
            self.step =  structure[2]
        else:
            self.step = 1

    def suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_int(
            self.name,
            self.low,
            self.high,
            self.step
        )
    
    def describe(self):
        return f"Integer, low: {self.low}, high: {self.high}, step: {self.step}"
    
class FloatParam(Param):

    def read_structure(self, structure: list):
        num_args = len(structure)
        assert num_args >= 3
        self.low = structure[0]
        self.high = structure[1]
        self.step =  structure[2]

    def suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_float(
            self.name,
            self.low,
            self.high,
            self.step
        )

    def describe(self):
        return f"Float, low: {self.low}, high: {self.high}, step: {self.step}"

class CatGenerator():

    def __init__(self,
                 apply: str,
                 args: DictConfig):
        self.apply = get_method(apply)
        self.args = args

    def get_comb(self) -> list[Any]:
        return self.apply(**self.args)

class CategoricalParam(Param):

    def read_structure(self, structure: list[Any] | CatGenerator):
        if isinstance(structure, CatGenerator):
            self.combs = structure.get_comb()
        else:
            self.combs = structure

    def suggest(self, trial: optuna.trial.Trial):
        return trial.suggest_categorical(
            self.name,
            self.combs
        )
    
    def describe(self):
        return f"Categorical, combinations number: {len(self.combs)}"

class OptimizerConfiguration:

    def __init__(self,
                 int_params: dict = {},
                 float_params: dict = {},
                 cat_params: dict = {}):
        self.params: list[Param] = [
            *[IntParam(name, structure) for name, structure in int_params.items()],
            *[FloatParam(name, structure) for name, structure in float_params.items()],
            *[CategoricalParam(name, structure) for name, structure in cat_params.items()]
        ]
        
    def add(self, param: Param):
        self.params.append(param)

    def describe(self):
        return {param.name: param.describe() for param in self.params}

def instantiate_optim_config(config: DictConfig) -> dict[str, OptimizerConfiguration]:
    return {
        key: instantiate(elem) for key, elem in config.optuna_config.items() 
        if isinstance(elem, OptimizerConfiguration) 
    }

class Optimizer():

    def __init__(self,
                 trainer_args: dict[str, Any],
                 model_class: Type[SFCN],
                 model_args: dict[str, Any],
                 trainable_args: dict[str, Any],
                 datamodule: LightningDataModule,
                 config: dict[OptimizerConfiguration],
                 logger: RankZeroLogger,
                 monitor_metric: str = "val_loss",
                 callbacks: list[Callback] = [],
                 ):
        self.default_trainer_args = trainer_args
        self.default_model_args = model_args
        self.default_trainable_args = trainable_args
        self.datamodule = datamodule
        self.config = config
        self.logger = logger
        self.monitor_metric = monitor_metric,
        self.callbacks = callbacks
        self.model_class = model_class

    def __register_parameters(arg_dict: dict[str, Any], 
                              suggested_parameters: dict[str, dict[str, Any]], 
                              key: str):
        if key in suggested_parameters:
            for param, val in suggested_parameters[key].items():
                arg_dict[param] = val # Register parameter in dict

    def __register_config(trial: optuna.trial.Trial, 
                          config: dict[str, OptimizerConfiguration]) -> dict[str, dict[str, Any]]:
        hyperparameters = {key: {} for key in config.keys()}
        for key, opt_config in config.items():
            for param in opt_config.params:
                value = param.suggest(trial)
                hyperparameters[key][param.name] = value
        return hyperparameters

    def optimize(self, n_trials=100, direction="minimize", timeout=600):
        
        def objective(trial: optuna.trial.Trial):

            model_args = deepcopy(self.default_model_args)
            trainer_args = deepcopy(self.default_trainer_args)
            trainable_args = deepcopy(self.default_trainable_args)

            suggested_parameters = self.__register_config(trial, self.config)
            self.__register_parameters(model_args, suggested_parameters, "model")
            self.__register_parameters(trainer_args, suggested_parameters, "trainer")
            self.__register_parameters(trainable_args, suggested_parameters, "trainable")

            callbacks = self.callbacks + [PyTorchLightningPruningCallback(trial, monitor=self.monitor_metric)]

            model: SFCN = self.model_class(**model_args)
            trainable: SFCNModule = SFCNModule(model=model, **trainable_args) 
            trainer = Trainer(callbacks=callbacks, **trainer_args)
            trainer.logger.log_hyperparams(suggested_parameters)

            self.logger.info(f"Starting trainer for trial")
            self.logger.info("Parameters:")
            for param, val in suggested_parameters:
                self.logger.info(f"-- {param}: {val}")
            trainer.fit(trainable, datamodule=self.datamodule)

            performances = trainer.callback_metrics
            self.logger.info(f"Finished. Total performance: {performances}")

            return trainer.callback_metrics[self.monitor_metric].item()

        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction=direction, pruner=pruner)
        self.logger.info("## Starting optimization ##")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.logger.info("Number of finished trials: {}".format(len(study.trials)))

        self.logger.info("Best trial:")
        trial = study.best_trial

        self.logger.info("  Value: {}".format(trial.value))

        self.logger.info("  Params: ")
        for key, value in trial.params.items():
            self.logger.info("    {}: {}".format(key, value))

# Utilities

def generate_combinations(filter_sizes):
    # List to hold all generated combinations
    combinations = []

    # Geometric progression (doubling and halving filters)
    combinations.append([filter_sizes[0] * (2 ** i) for i in range(len(filter_sizes)) if filter_sizes[0] * (2 ** i) <= max(filter_sizes)])
    combinations.append([filter_sizes[0] // (2 ** i) for i in range(len(filter_sizes)) if filter_sizes[0] // (2 ** i) >= min(filter_sizes)])

    # Increasing then decreasing progression
    for i in range(len(filter_sizes)):
        increasing = filter_sizes[:i+1]
        decreasing = filter_sizes[i:][::-1]
        combinations.append(increasing + decreasing)

    # Constant filters (same filter size throughout)
    for size in filter_sizes:
        combinations.append([size] * len(filter_sizes))

    # Small to large progression
    for i in range(len(filter_sizes)):
        small_to_large = filter_sizes[:i+1] + sorted(filter_sizes[i+1:], reverse=True)
        combinations.append(small_to_large)

    # Large to small progression
    for i in range(len(filter_sizes)):
        large_to_small = filter_sizes[:i+1] + sorted(filter_sizes[i+1:])
        combinations.append(large_to_small)

    # Randomized combinations (for a large number of combinations)
    random_combinations = itertools.product(filter_sizes, repeat=len(filter_sizes))
    combinations.extend(random_combinations)

    # Filter combinations with increasing-decreasing complexity
    combinations.append([min(filter_sizes)] + sorted(filter_sizes[1:], reverse=True))
    combinations.append([max(filter_sizes)] + sorted(filter_sizes[1:]))

    # Clean-up duplicates by converting to a set and back to a list
    unique_combinations = [list(x) for x in set(tuple(x) for x in combinations)]

    return unique_combinations