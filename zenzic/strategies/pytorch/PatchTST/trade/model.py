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

# PatchTST in Trades Classifier.
class PatchTSTTrades(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

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

        self.flaten = nn.Flatten()
        self.linear = nn.Linear(in_features=configs.enc_in*configs.pred_len, out_features=1, bias=False)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        # self.linear.bias.data.fill_(0.01)
        self.activation = nn.Sigmoid()
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

    def forward(self, x):
        backbone_out = self.backbone(x)
        flatten_out = self.flaten(backbone_out)
        linear_out = self.linear(flatten_out)
        return self.activation(torch.squeeze(linear_out))
    
    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y, 'binary')
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
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
        parser.add_argument('--seq_len', type=int, default=256, help='Input sequence length.')
        parser.add_argument('--pred_len', type=int, default=256, help='Prediction sequence length.')
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
        parser.add_argument('--decomposition', type=bool, default=True, help='If true, decompose input into trend and residual.')
        parser.add_argument('--kernel_size', type=int, default=21, help='The window size of moving average when decompose.')

        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-4)
        parser.add_argument('--lr_patience', type=int, default=5)

        return parser
