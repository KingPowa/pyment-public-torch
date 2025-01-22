import unittest
import optuna
import yaml
from omegaconf import OmegaConf
from unittest.mock import MagicMock
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.callbacks import Callback

from pyment.utils.optimize import Optimizer, OptimizerConfiguration, IntParam, FloatParam, RankZeroLogger


class DummyModel(LightningModule):
    def __init__(self, lr=0.001, hidden_size=128):
        super().__init__()
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        return 0.5

    def configure_optimizers(self):
        return None


class DummyDataModule(LightningDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return []


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up common test data before each test."""
        trainer_args = {"max_epochs": 10}
        model_args = {"lr": 0.001, "hidden_size": 128}
        datamodule = DummyDataModule()
        logger = MagicMock(spec=RankZeroLogger)

        config = {
            "model": OptimizerConfiguration(
                int_param={"hidden_size": [64, 256, 32]},
                float_param={"lr": [0.0001, 0.01, 0.0001]}
            ),
            "trainer": OptimizerConfiguration(
                int_param={"max_epochs": [5, 20, 5]},
                float_param={}
            )
        }

        self.optimizer = Optimizer(
            trainer_args=trainer_args,
            model_class=DummyModel,
            model_args=model_args,
            datamodule=datamodule,
            config=config,
            logger=logger
        )

    def test_int_param(self):
        param = IntParam("hidden_size", [64, 256, 32])
        self.assertEqual(param.low, 64)
        self.assertEqual(param.high, 256)
        self.assertEqual(param.step, 32)

    def test_float_param(self):
        param = FloatParam("lr", [0.0001, 0.01, 0.0001])
        self.assertEqual(param.low, 0.0001)
        self.assertEqual(param.high, 0.01)
        self.assertEqual(param.step, 0.0001)

    def test_register_config(self):
        trial = MagicMock(spec=optuna.trial.Trial)
        trial.suggest_int.side_effect = lambda name, low, high, step: low
        trial.suggest_float.side_effect = lambda name, low, high, step: low

        result = self.optimizer._Optimizer__register_config(trial, self.optimizer.config)

        self.assertIn("model", result)
        self.assertIn("trainer", result)
        self.assertIn("hidden_size", result["model"])
        self.assertIn("lr", result["model"])
        self.assertIn("max_epochs", result["trainer"])
        self.assertEqual(result["model"]["hidden_size"], 64)
        self.assertEqual(result["model"]["lr"], 0.0001)
        self.assertEqual(result["trainer"]["max_epochs"], 5)

    def test_register_parameters(self):
        model_args = {}
        suggested_parameters = {
            "model": {"hidden_size": 128, "lr": 0.001},
            "trainer": {"max_epochs": 10}
        }

        self.optimizer._Optimizer__register_parameters(model_args, suggested_parameters, "model")
        self.assertEqual(model_args["hidden_size"], 128)
        self.assertEqual(model_args["lr"], 0.001)

    def test_optimize_function(self):
        """Runs optimization with a single trial to verify execution."""
        self.optimizer.optimize(n_trials=1, timeout=10)  # Single trial test
        self.optimizer.logger.info.assert_called()

    def test_yaml_loading_and_configuration_parsing(self):
        yaml_content = """
        model:
            int_param:
                hidden_size: [64, 256, 32]
            float_param:
                lr: [0.0001, 0.01, 0.0001]

        trainer:
            int_param:
                max_epochs: [5, 20, 5]
            float_param: {}
        """

        yaml_dict = yaml.safe_load(yaml_content)
        config = OmegaConf.create(yaml_dict)

        optimizer_configs = {
            key: OptimizerConfiguration(
                int_param=config[key].get("int_param", {}),
                float_param=config[key].get("float_param", {})
            ) for key in config.keys()
        }

        self.assertIn("model", optimizer_configs)
        self.assertIn("trainer", optimizer_configs)
        self.assertIsInstance(optimizer_configs["model"], OptimizerConfiguration)
        self.assertIsInstance(optimizer_configs["trainer"], OptimizerConfiguration)
        self.assertEqual(optimizer_configs["model"].params[0].name, "hidden_size")
        self.assertEqual(optimizer_configs["model"].params[1].name, "lr")


if __name__ == "__main__":
    unittest.main()