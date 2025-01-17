import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from .sfcn import SFCN

class SFCNModule(pl.LightningModule):
    def __init__(self, model: SFCN, learning_rate=0.1, optimizer = "sgd", criterion = nn.MSELoss()):
        """
        PyTorch Lightning module to train SFCN
        
        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            learning_rate (float): Initial learning rate for SGD (default: 0.1).
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.model(img)
        loss = self.criterion(predicted_age, cond[0])
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.model(img)
        loss = self.criterion(predicted_age, cond[0])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, cond = batch
        predicted_age = self.model(img)
        loss = self.criterion(predicted_age, cond[0])
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def __get_opt(self, opt):
        if opt == "sgd": return optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else: raise NotImplementedError(f"Optimizer {opt} not supported.")

    def configure_optimizers(self):
        """
        Configure SGD optimizer with a step-wise learning rate schedule.
        - Initial learning rate: self.learning_rate
        - Step-wise reduction by factor of 3 at epochs [20, 40, 60]
        """
        optimizer = self.__get_opt(self.optimizer)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=1/3)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}