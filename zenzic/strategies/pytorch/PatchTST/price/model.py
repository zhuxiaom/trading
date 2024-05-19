import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import zenzic.thirdparty.PatchTST.PatchTST_supervised.models.PatchTST as PatchTST
from madgrad import MADGRAD
from torch.optim import Adam
from torchmetrics.functional import accuracy, f1_score, precision, recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser
from torchmetrics.functional.regression import (
    mean_absolute_error,
    # mean_absolute_percentage_error,
    mean_squared_error,
    # mean_squared_log_error,
)

def daily_profit(cum_profit):
    p = torch.log(torch.clamp(cum_profit, min=1e-10))
    p[:, 1:] = p[:, 1:] - p[:, 0:-1]
    return torch.exp(p)

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

# PatchTST in Prices Regressor.
class PatchTSTPrices(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        # PatchTST as backbone.
        self.backbone = PatchTST.Model(
            configs,
            # max_seq_len = configs.max_seq_len,
            d_k = configs.d_k,
            d_v = configs.d_v,
            # norm = configs.norm,
            attn_dropout = configs.attn_dropout,
            act = configs.act,
            # key_padding_mask = configs.key_padding_mask,
            # padding_var = configs.padding_var,
            # attn_mask = configs.attn_mask,
            res_attention = configs.res_attention,
            pre_norm = configs.pre_norm,
            store_attn = configs.store_attn,
            pe = configs.pe,
            learn_pe = configs.learn_pe,
        )

        self.output = nn.Linear(in_features=configs.enc_in, out_features=1, bias=False)
        # self.register_buffer('loss_factor', torch.Tensor([100.0]))

    def __resize_data(self, x):
        x = x[:, -self.configs.seq_len:, -self.configs.enc_in:]
        return x

    def __metrics(self, stage: str, y_hat, y):
        # assert not torch.any(y_hat < -9e-8), f"Found invalid value! {torch.min(y_hat)}"
        loss = mean_squared_error(y_hat, y)
        # calculate other metrics.
        # y_ret = torch.special.logit(y_hat / self.configs.loss_scale, eps=1e-8)
        cum_mae = mean_absolute_error(y_hat, y)
        r = daily_profit(y)
        r_pred = daily_profit(y_hat)
        mae = mean_absolute_error(r_pred, r)
        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_mae': mae,
            f'{stage}_cum_mae': cum_mae,
        }, on_epoch=True, on_step=False)
        return loss

    def forward(self, x):   # x.shape is (batch, seq_len, c_in)
        backbone_out = self.backbone(x)
        return self.output(backbone_out).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        # y = y[:, :, 3]
        y_hat = self(x)
        return self.__metrics('train', y_hat=y_hat, y=y)
    
    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        # y = y[:, :, 3]
        y_hat = self(x)
        self.__metrics('val', y_hat=y_hat, y=y)

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        # y = y[:, :, 3]
        y_hat = self(x)
        self.__metrics('test', y_hat=y_hat, y=y)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), self.configs.learning_rate)
        # optimizer = Adam(self.parameters(), self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=self.configs.lr_patience,
            cooldown=self.configs.lr_patience / 2, factor=0.5)
        if self.configs.lr_patience <= 0:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss"
            }

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        ### -------  model settings --------------
        # parser.add_argument('--max_seq_len', type=int, default=256, help='Max length of time series.')
        parser.add_argument('--d_k', type=int, default=None, help='The dimension of Attention keys.')
        parser.add_argument('--d_v', type=int, default=None, help='The dimension of Attention values.')
        parser.add_argument('--attn_dropout', type=float, default=0.0, help='The dropout ratio of Attention layer.')
        parser.add_argument('--act', type=str, default='gelu', help='The activation func of Attention (relu or gelu).')
        parser.add_argument('--res_attention', type=bool, default=False, help='If true, enable residual attention.')
        parser.add_argument('--pre_norm', type=bool, default=False, help='If true, enable pre-normalization of Attention.')
        parser.add_argument('--store_attn', type=bool, default=False, help='If true, store attention.')
        parser.add_argument('--pe', type=str, default='zeros', help='The position embedding strategy.')
        parser.add_argument('--learn_pe', type=bool, default=True, help='If true, enable the learning of position embeddings.') #TODO: try LoPE
        parser.add_argument('--enc_in', type=int, default=4, help='Encoder input dimension.')
        parser.add_argument('--seq_len', type=int, default=384, help='Input sequence length.')
        parser.add_argument('--pred_len', type=int, default=24, help='Prediction sequence length.')
        parser.add_argument('--e_layers', type=int, default=3, help='The number of Attention layers.')
        parser.add_argument('--n_heads', type=int, default=16, help='The number of Attention heads.')
        parser.add_argument('--d_model', type=int, default=128, help='The model dimension.')
        parser.add_argument('--d_ff', type=int, default=256, help='The feedforward dimension.')
        parser.add_argument('--dropout', type=float, default=0.0, help='The dropout of input.')
        parser.add_argument('--fc_dropout', type=float, default=0.0, help='The dropout of pre-trained head.')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='The dropout of Attention head.')
        parser.add_argument('--individual', type=bool, default=False, help='Individual head of each channel.')
        parser.add_argument('--patch_len', type=int, default=32, help='The patch length.')
        parser.add_argument('--stride', type=int, default=1, help='The stride of patching.')
        parser.add_argument('--padding_patch', type=str, default=None, help='Whether to pad at the end (None or end).')
        parser.add_argument('--revin', type=bool, default=True, help='If true, enable Reversible Instance Normalization (RevIN).')
        parser.add_argument('--affine', type=bool, default=True, help='If true, RevIN has learnable affine parameters.')
        parser.add_argument('--subtract_last', type=bool, default=False, help='If true, substract last in RevIN.')
        parser.add_argument('--decomposition', type=bool, default=False, help='If true, decompose input into trend and residual.')
        parser.add_argument('--kernel_size', type=int, default=21, help='The window size of moving average when decompose.')

        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-4)
        parser.add_argument('--lr_patience', type=int, default=5)

        return parser
