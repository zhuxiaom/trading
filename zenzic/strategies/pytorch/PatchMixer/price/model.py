import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn

# import torch.nn.functional as F
from zenzic.thirdparty.PatchMixer.models import PatchMixer
from madgrad import MADGRAD
from torch.optim import Adam
from torchmetrics.functional.regression import (
    mean_absolute_error,
    # mean_absolute_percentage_error,
    mean_squared_error,
    # mean_squared_log_error,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from lightning.pytorch.utilities import grad_norm
from zenzic.strategies.pytorch.common import act_funcs
from zenzic.strategies.pytorch.common.logistic_layer import LogisticLayer
from zenzic.strategies.pytorch.common.mask_layer import MaskLayer

def normalize(x, y):
    x = torch.log(x)
    x[:, 0:1, :4] = x[:, 0:1, :4] - x[:, 0:1, 3:4].expand(x.shape[0], 1, 4)
    x[:, 1:, :4] = x[:, 1:, :4] - x[:, 0:-1, :4]
    x[:, 0:1, 4:] = x[:, 0:1, 4:] - x[:, 0:1, 7:8].expand(x.shape[0], 1, 4)
    x[:, 1:, 4:] = x[:, 1:, 4:] - x[:, 0:-1, 4:]

    y = torch.log(y)
    y[:, 1:] = y[:, 1:] - y[:, 0:-1]
    # assert not torch.any(torch.isnan(x)), f"Found nan value in X when normalize:\n{x[torch.isnan(x)]}"
    # assert not torch.any(torch.isnan(y)), f"Found nan value in Y when normalize:\n{y[torch.isnan(y)]}"
    return x, y

def denormalize(y):
    y_ret = torch.cumsum(y, dim=1)
    y_ret = torch.exp(y_ret)
    # assert not torch.any(torch.isnan(y)), f"Found nan value in Y when denormalize: {torch.min(y)}, {torch.max(y)}"
    return y_ret


# PatchMixer to predict prices.
class PatchMixerPrice(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()

        self.configs = configs
        
        # PatchMixer as backbone.
        self.mask = MaskLayer(shape=(self.configs.seq_len, self.configs.enc_in))
        self.logistic = LogisticLayer(self.configs.enc_in)
        self.backbone = PatchMixer.Model(configs)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_features=configs.enc_in, out_features=1)
        )

        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience
        self.register_buffer('loss_factor', torch.Tensor([100.0]))

    def __resize_data(self, x):
        x = x[:, -self.configs.seq_len:, -self.configs.enc_in:]
        return x
    
    def __metrics(self, stage: str, y_hat, y):
        # assert not torch.any(y_hat < -9e-8), f"Found invalid value! {torch.min(y_hat)}"
        loss = mean_squared_error(y_hat, y) * self.loss_factor
        # calculate other metrics.
        # y_ret = torch.special.logit(y_hat / self.configs.loss_scale, eps=1e-8)
        y_ret = y_hat
        r = torch.exp(y)
        r_pred = torch.exp(y_ret)
        mae = mean_absolute_error(r_pred, r)
        cum_r = denormalize(y)
        cum_r_pred = denormalize(y_ret)
        cum_mae = mean_absolute_error(cum_r_pred, cum_r)
        if stage == 'train':
            self.log('feature_count', self.mask.activated_count, on_epoch=True, on_step=False)
        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_mae': mae,
            f'{stage}_cum_mae': cum_mae,
        }, on_epoch=True, on_step=False)
        return loss

    def forward(self, x):
        logi_x = self.logistic(x)
        mask_x = self.mask(logi_x)
        backbone_out = self.backbone(mask_x)
        # assert not torch.any(backbone_out < -9e-8), f"Found invalid value! {torch.min(backbone_out)}"
        out = self.output(backbone_out)
        return torch.squeeze(out)
    
    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        return self.__metrics('train', y_hat=y_hat, y=y)
    
    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        self.__metrics('val', y_hat=y_hat, y=y)

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        self.__metrics('test', y_hat=y_hat, y=y)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), self.learning_rate)
        # optimizer = Adam(self.parameters(), self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=self.lr_patience,
            cooldown=self.lr_patience / 2, factor=0.1, threshold=self.learning_rate/100)
        if self.lr_patience <= 0:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }

    # def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self.backbone, norm_type=2)
        # self.log_dict(norms)

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        ### -------  model settings --------------
        parser.add_argument('--seq_len', type=int, default=384, help='Input sequence length.')
        parser.add_argument('--pred_len', type=int, default=24, help='Output sequence length of PatchMixer block.')
        parser.add_argument('--enc_in', type=int, default=8, help='Num of input features (channels).')
        parser.add_argument('--d_model', type=int, default=128, help='The model dimension (num of features).')
        parser.add_argument('--patch_len', type=int, default=5, help='The length of patch.')
        parser.add_argument('--stride', type=int, default=1, help='The stride of patch.')
        parser.add_argument('--mixer_kernel_size', type=int, default=5, help='The mixer kernel size.')
        parser.add_argument('--dropout', type=float, default=0.5, help='The dropout ratio.')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='The head dropout ratio.')
        parser.add_argument('--e_layers', type=int, default=6, help='The number of PatchMixer layers.')

        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-5)
        parser.add_argument('--lr_patience', type=int, default=5)

        return parser
