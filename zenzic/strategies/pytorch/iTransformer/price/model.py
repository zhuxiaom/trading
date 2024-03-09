import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
# import torch.nn.functional as F
from zenzic.thirdparty.TimeSeriesLibrary.models import iTransformer
from madgrad import MADGRAD
from torch.optim import Adam
from torchmetrics.functional.regression import mean_absolute_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, mean_squared_log_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from lightning.pytorch.utilities import grad_norm

def normalize(x, y):
    x = torch.log(x)
    x[:, 0:1, :4] = x[:, 0:1, :4] - x[:, 0:1, 3:4].expand(x.shape[0], 1, 4)
    x[:, 1:, :4] = x[:, 1:, :4] - x[:, 0:-1, :4]
    x[:, 0:1, 4:] = x[:, 0:1, 4:] - x[:, 0:1, 7:8].expand(x.shape[0], 1, 4)
    x[:, 1:, 4:] = x[:, 1:, 4:] - x[:, 0:-1, 4:]

    y = torch.log(y)
    y[:, 1:] = y[:, 1:] - y[:, 0:-1]
    assert not torch.any(torch.isnan(x)), f"Found nan value in X when normalize:\n{x[torch.isnan(x)]}"
    assert not torch.any(torch.isnan(y)), f"Found nan value in Y when normalize:\n{y[torch.isnan(y)]}"
    return x, y

def denormalize(y):
    y = torch.cumsum(y, dim=1)
    y = torch.exp(y)
    assert not torch.any(torch.isnan(y)), f"Found nan value in Y when denormalize:\n{y[torch.isnan(y)]}"
    return y


# iTransformer to predict prices.
class iTransPrice(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()

        self.configs = configs

        # iTransformer as backbone.
        self.backbone = iTransformer.Model(configs)
        self.output = nn.Linear(configs.c_in, 1)

        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

    def __resize_data(self, x):
        x = x[:, -self.configs.seq_len:, -self.configs.c_in:]
        return x

    def forward(self, x):
        # x_mark = torch.ones(x.shape[0], self.configs.seq_len).to(x.device)
        backbone_out = self.backbone(x, None, None, None)
        return self.output(backbone_out).squeeze_()
    
    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        loss = mean_squared_log_error(y_hat, y)

        y_hat = denormalize(y_hat)
        y = denormalize(y)
        mae = mean_absolute_error(y_hat, y)
        mape = mean_absolute_percentage_error(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True)
        self.log('train_mape', mape, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        # print(f'y_hat: {y_hat.shape}, y: {y.shape}')
        loss = mean_squared_log_error(y_hat, y)

        y_hat = denormalize(y_hat)
        y = denormalize(y)
        mae = mean_absolute_error(y_hat, y)
        mape = mean_absolute_percentage_error(y_hat, y)
        self.log_dict({
            'val_loss': loss,
            'val_mae': mae,
            'val_mape': mape,
        }, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y = y[:, :, 3]
        x, y = normalize(x, y)
        y_hat = self(x)
        loss = mean_squared_log_error(y_hat, y)

        y_hat = denormalize(y_hat)
        y = denormalize(y)
        mae = mean_absolute_error(y_hat, y)
        mape = mean_absolute_percentage_error(y_hat, y)
        self.log_dict({
            'test_loss': loss,
            'test_mae': mae,
            'test_mape': mape,
        }, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), self.learning_rate)
        # optimizer = Adam(self.parameters(), self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=self.lr_patience,
            cooldown=self.lr_patience / 2, factor=0.5)
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
        parser.add_argument('--task_name', type=str, default='long_term_forecast', help='Specify the task as time series forcast.')
        parser.add_argument('--seq_len', type=int, default=384, help='Input sequence length.')
        parser.add_argument('--pred_len', type=int, default=24, help='Output sequence length of iTransformer block.')
        parser.add_argument('--output_attention', type=bool, default=False, help='Whether to output attention weights.')
        parser.add_argument('--c_in', type=int, default=8, help='Num of input features (channels).')
        parser.add_argument('--d_model', type=int, default=32, help='The model dimension (num of features).')
        parser.add_argument('--embed', type=str, default=None, help='The type of embedding (not used).')
        parser.add_argument('--freq', type=str, default=None, help='The frequency of the time (not used).')
        parser.add_argument('--dropout', type=float, default=0.5, help='The dropout ratio.')
        parser.add_argument('--factor', type=int, default=0, help='The factor (not used).')
        parser.add_argument('--n_heads', type=int, default=8, help='The num of attention heads.')
        parser.add_argument('--d_ff', type=int, default=512, help='The feedforward dimension.')
        parser.add_argument('--activation', type=str, default='gelu', help='The activation function (gelu or relu).')
        parser.add_argument('--e_layers', type=int, default=3, help='The number of iTransformer layers.')

        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-4)
        parser.add_argument('--lr_patience', type=int, default=5)

        return parser
