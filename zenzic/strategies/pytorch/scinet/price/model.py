from unicodedata import bidirectional
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy, f1_score, precision_recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zenzic.thirdparty.SCINet.models.SCINet import SCINet
from zenzic.thirdparty.SCINet.models.SCINet_decompose import SCINet_decompose
from argparse import ArgumentParser

# SCINet in Trades Classifier.


class SCINetTrades(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        # Forward backbone network.
        self.backbone = None
        if not self.configs.decomponse:
            self.backbone = SCINet(
                output_len=self.configs.output_len,
                input_len=self.configs.window_size,
                input_dim=self.configs.input_rnn_hidden_size,
                hid_size=self.configs.hidden_size,
                num_stacks=self.configs.stacks,
                num_levels=self.configs.levels,
                num_decoder_layer=self.configs.num_decoder_layer,
                concat_len=self.configs.concat_len,
                groups=self.configs.groups,
                kernel=self.configs.kernel,
                dropout=self.configs.dropout,
                single_step_output_One=self.configs.single_step_output_One,
                positionalE=self.configs.positionalEcoding,
                modified=True,
                RIN=self.configs.RIN
            )
        else:
            self.backbone = SCINet_decompose(
                output_len=self.configs.horizon,
                input_len=self.configs.window_size,
                input_dim=self.configs.input_dim,
                hid_size=self.configs.hidden_size,
                num_stacks=self.configs.stacks,
                num_levels=self.configs.levels,
                # num_decoder_layer=configs.num_decoder_layer,
                concat_len=self.configs.concat_len,
                groups=self.configs.groups,
                kernel=self.configs.kernel,
                dropout=self.configs.dropout,
                single_step_output_One=self.configs.single_step_output_One,
                positionalE=self.configs.positionalEcoding,
                modified=True,
                RIN=self.configs.RIN
            )
        self.out_projection = nn.Conv1d(self.configs.input_dim, 1, 1, 1, bias=False)

    def forward(self, x):
        # calculate log return
        _, seq_len, _ = x.shape
        close_price = x[:, :, 3:4].clone().detach()
        x_in = x.clone().detach()
        x_in[:, 1:, :] = torch.log(x_in[:, 1:, :] / x[:, :seq_len-1, :])
        x_in[:, :1, :4] = torch.log(x_in[:, :1, :4]/close_price.repeat(1, 1, 4))
        x_in[:, :1, 4:] = torch.zeros_like(x_in[:, :1, 4:])
        pass

    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat = self(x)
        # print(f'y_hat: {y_hat.shape}, y: {y.shape}')
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        f1 = f1_score(y_hat, y)
        prec_recall = precision_recall(y_hat, y)
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
            'val_prec': prec_recall[0],
            'val_recall': prec_recall[1]
        })

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        f1 = f1_score(y_hat, y)
        prec_recall = precision_recall(y_hat, y)
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc,
            'test_f1': f1,
            'test_prec': prec_recall[0],
            'test_recall': prec_recall[1]
        })

    def configure_optimizers(self):
        # optimizer = MADGRAD(self.parameters(), self.learning_rate)
        optimizer = Adam(self.parameters(), self.learning_rate)
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

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # -------  model settings --------------
        parser.add_argument('--hidden_size', default=2, type=float,
                            help='hidden channel of module')  # H, EXPANSION RATE
        parser.add_argument('--INN', default=1, type=int,
                            help='use INN or basic strategy')
        parser.add_argument('--kernel', default=5, type=int,
                            help='kernel size')  # k kernel size
        parser.add_argument('--dilation', default=1, type=int, help='dilation')
        parser.add_argument('--positionalEcoding', type=bool, default=False)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--levels', type=int, default=3)
        parser.add_argument('--num_decoder_layer', type=int, default=1)
        parser.add_argument('--stacks', type=int, default=1)
        parser.add_argument('--RIN', type=bool, default=False)
        parser.add_argument('--input_rnn_hidden_size', type=int, default=4)
        parser.add_argument('--decompose', type=bool,
                            default=False, help='Enable SCINet Decompose model.')
        # -------  input/output length settings --------------
        parser.add_argument('--window_size', type=int,
                            default=96, help='input length')
        parser.add_argument('--input_dim', type=int,
                            default=5, help='input length')
        parser.add_argument('--output_len', type=int,
                            default=1, help='output length')
        parser.add_argument('--concat_len', type=int, default=165)
        parser.add_argument('--single_step_output_One', type=int,
                            default=0, help='only output the single final step')
        # -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--lr_patience', type=int, default=10)

        return parser
