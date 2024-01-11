from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn.modules as nn
import torch
from pytorch_lightning import LightningModule


# implementation of simple U-net with 2 downsampling steps
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # downsample blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(12,64,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding='same'),
            nn.ReLU(),
        )
        self.down1 = nn.MaxPool2d(2) # 64x48x48
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128,128,3, padding='same'),
            nn.ReLU(),
        )
        self.down2 = nn.MaxPool2d(2) # 128x24x24

        self.block3 = nn.Sequential(
            nn.Conv2d(128,128,3, padding='same'),
            nn.ReLU()
        )

        # upsample blocks
        self.up2 = nn.Upsample(scale_factor=2) # 128x48x48
        self.block2_u = nn.Sequential(
            nn.Conv2d(256,128,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128,64,3, padding='same'), 
            nn.ReLU()
        )
        self.up1 = nn.Upsample(scale_factor=2) # 64x96x96
        self.block1_u = nn.Sequential(
            nn.Conv2d(128,64,3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding='same'), 
            nn.ReLU()
        )
        self.block_out = nn.Conv2d(64,1,1)
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
        return self.block_out(b1_up)
    
class Pl_wrapper(LightningModule):

    def __init__(self, pytorch_model):
        super().__init__()
        self.model = pytorch_model
        self.loss_fct = nn.CrossEntropyLoss()

    def training_step(self, dls, batch_idx):
        loss = 0
        for batch in dls:
            s2, target, _ = batch
            preds = self.model(s2)
            loss += self.loss_fct(preds, target)
        loss = loss / len(dls)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def validation_step(self, batch, *args, **kwargs):
        s2, target, _ = batch
        preds = self.model(s2)
        loss = self.loss_fct(preds, target)
        self.log('val_loss', loss)