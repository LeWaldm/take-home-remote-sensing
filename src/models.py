from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn.modules as nn
import torch
from pytorch_lightning import LightningModule


class UNet(nn.Module):
    """ Simple U-Net with 12x96x96 expected input and 2 up/down-sample blocks"""
    def __init__(self):
        super().__init__()

        # downsample blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(12,16,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,16,3, padding='same'),
            nn.ReLU(),
        )
        self.down1 = nn.MaxPool2d(2) # 16x48x48
        self.block2 = nn.Sequential(
            nn.Conv2d(16,32,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32,32,3, padding='same'),
            nn.ReLU(),
        )
        self.down2 = nn.MaxPool2d(2) # 32x24x24

        self.block3 = nn.Sequential(
            nn.Conv2d(32,32,3, padding='same'),
            nn.ReLU()
        )

        # upsample blocks
        self.up2 = nn.Upsample(scale_factor=2) # 32x48x48
        self.block2_u = nn.Sequential(
            nn.Conv2d(64,32,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32,16,3, padding='same'), 
            nn.ReLU()
        )
        self.up1 = nn.Upsample(scale_factor=2) # 16x96x96
        self.block1_u = nn.Sequential(
            nn.Conv2d(32,16,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16,16,3, padding='same'), 
            nn.ReLU()
        )
        self.block_out = nn.Conv2d(16,1,1)
        self.float()

    def forward(self, x):

        # downsample
        b1 = self.block1(x)
        b2 = self.block2(self.down1(b1))
        b3 = self.block3(self.down2(b2))

        # upsample
        b2_up = torch.cat( [b2,self.up2(b3)], axis=1)
        b2_up = self.block2_u(b2_up)
        b1_up = torch.cat( [b1,self.up1(b2_up)], axis=1)
        b1_up = self.block1_u(b1_up)

        # output dimension
        return self.block_out(b1_up).squeeze(dim=1)
    
class Pl_wrapper(LightningModule):
    """ Simple Pytorch-lightning wrapper for any pytorch model """

    def __init__(self, pytorch_model):
        super().__init__()
        self.model = pytorch_model
        self.loss_fct = nn.BCELoss()
        self.sigm = nn.Sigmoid()

    def training_step(self, batch, batch_idx):
        s2, target, _ = batch
        preds = self.model(s2)
        loss = self.loss_fct(self.sigm(preds), target)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch, *args, **kwargs):
        s2, target, _ = batch
        preds = self.model(s2)
        loss = self.loss_fct(preds, target)
        self.log('val_loss', loss)