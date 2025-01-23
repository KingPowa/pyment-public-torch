import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from logging import Logger
from .sfcn import SFCN

class SFCNModule(pl.LightningModule):
    def __init__(self, eff_model: SFCN, 
                 optimizer = "sgd", 
                 criterion = nn.MSELoss(), 
                 max_epochs = 80, 
                 decay = 1/3, 
                 milestones = [20,40,60],
                 lr = 0.1,
                 weight_decay = 0,
                 momentum = 0,
                 pers_logger: Logger = None):
        """
        PyTorch Lightning module to train SFCN
        
        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            learning_rate (float): Initial learning rate for SGD (default: 0.1).
        """
        super().__init__()
        self.eff_model = eff_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.pers_logger = pers_logger
        self.save_hyperparameters(ignore=["criterion", "pers_logger"])

    def forward(self, x):
        return self.eff_model(x)

    def training_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.eff_model(img)
        loss = self.criterion(predicted_age, cond[:, 0].unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.eff_model(img)
        loss = self.criterion(predicted_age, cond[:, 0].unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.eff_model(img)
        loss = self.criterion(predicted_age, cond[:, 0].unsqueeze(1))
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("train_loss", "N/A")
        self.pers_logger.info(f"End of Epoch {self.current_epoch}: Avg Train Loss = {avg_loss}")

        # Manually flush logs to ensure order is preserved
        for handler in self.pers_logger.handlers:
            handler.flush()

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("val_loss", "N/A")
        self.pers_logger.info(f"End of Epoch {self.current_epoch}: Avg Validation Loss = {avg_loss}")

        # Manually flush logs to ensure order is preserved
        for handler in self.pers_logger.handlers:
            handler.flush()

    def __get_opt(self, opt):
        if opt == "sgd": return optim.SGD(self.eff_model.parameters(), 
                                          lr=self.hparams.lr,
                                          weight_decay=self.hparams.weight_decay,
                                          momentum=self.hparams.momentum)
        else: raise NotImplementedError(f"Optimizer {opt} not supported.")

    def configure_optimizers(self):
        """
        Configure SGD optimizer with a step-wise learning rate schedule.
        - Initial learning rate: self.lr
        - Step-wise reduction by factor of 3 at epochs [20, 40, 60]
        """
        optimizer = self.__get_opt(self.hparams.optimizer)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.decay)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}