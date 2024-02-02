import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from zenzic.thirdparty.TimeSeriesLibrary.models import TimesNet
from madgrad import MADGRAD
from torch.optim import Adam
from torchmetrics.functional import accuracy, f1_score, precision, recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argparse import ArgumentParser

# TimesNet in Trades Classifier.
class TimesNetTrades(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()

        self.configs = configs

        # TimesNet as backbone.
        self.backbone = TimesNet.Model(configs)

        self.sigmoid = nn.Sigmoid()
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

    def __resize_data(self, x):
        x = x[:, -self.configs.seq_len:, -self.configs.enc_in:]
        return x

    def forward(self, x):
        x_mark = torch.ones(x.shape[0], self.configs.seq_len).to(x.device)
        backbone_out = self.backbone(x, x_mark, None, None)
        return self.sigmoid(backbone_out.squeeze())
    
    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y, 'binary')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y_hat = self(x)
        # print(f'y_hat: {y_hat.shape}, y: {y.shape}')
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y, 'binary')
        f1 = f1_score(y_hat, y, 'binary')
        prec = precision(y_hat, y, 'binary')
        rec = recall(y_hat, y, 'binary')
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
            'val_prec': prec,
            'val_recall': rec
        }, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        x = self.__resize_data(x)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y, 'binary')
        f1 = f1_score(y_hat, y, 'binary')
        prec = precision(y_hat, y, 'binary')
        rec = recall(y_hat, y, 'binary')
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc,
            'test_f1': f1,
            'test_prec': prec,
            'test_recall': rec
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

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        ### -------  model settings --------------
        parser.add_argument('--task_name', type=str, default='classification', help='Specify the task as classification.')
        parser.add_argument('--seq_len', type=int, default=128, help='Input sequence length.')
        parser.add_argument('--pred_len', type=int, default=0, help='Output sequence length of TimesNet block.')    # Not used
        parser.add_argument('--num_class', type=int, default=1, help='The num of classes.')
        parser.add_argument('--label_len', type=int, default=1, help='The num of labels.')  # Not used
        parser.add_argument('--top_k', type=int, default=5, help='Top K of FFT frequencies.')
        parser.add_argument('--d_model', type=int, default=32, help='The model dimension.')
        parser.add_argument('--d_ff', type=int, default=32, help='The feedforward dimension.')
        parser.add_argument('--num_kernels', type=int, default=10, help='The num of kernels in Inception block.')
        parser.add_argument('--e_layers', type=int, default=3, help='The number of TimesNet layers.')
        parser.add_argument('--dropout', type=float, default=0.0, help='The dropout ratio.')
        parser.add_argument('--enc_in', type=int, default=4, help='Encoder input dimension.')
        parser.add_argument('--embed', type=str, default='fixed', help='The position embedding strategy.')
        parser.add_argument('--freq', type=str, default='h', help='The position embedding strategy.')
        parser.add_argument('--norm_mode', type=int, default=0, help='If 0, use nn.LayerNorm; if 1, use RevIN but norm only; if 2, use RevIN norm and denorm.')
        parser.add_argument('--act_func', type=str, default='tanh', help='Activation function (either tanh or gelu).')

        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-4)
        parser.add_argument('--lr_patience', type=int, default=5)

        return parser
