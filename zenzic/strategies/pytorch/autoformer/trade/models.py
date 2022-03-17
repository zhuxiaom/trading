from distutils.command.config import config
from email.policy import default
from sched import scheduler
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from zenzic.thirdparty.Autoformer.layers.Embed import DataEmbedding_wo_pos
from zenzic.thirdparty.Autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from zenzic.thirdparty.Autoformer.layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm
from zenzic.thirdparty.Autoformer.models import Autoformer
from madgrad import MADGRAD
from torchmetrics.functional import accuracy, f1_score, precision_recall
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Autoformer Encoder of Trades model
class AEoT(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )

        # Output
        self.dense = nn.Linear(configs.d_model, out_features=1, bias=True)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        self.activation = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # add cls to the end.
        cls = torch.zeros(
            enc_out.shape[0], 1, enc_out.shape[2], device=enc_out.device)
        enc_out = torch.cat((enc_out, cls), dim=1)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # output
        cls_in = enc_out[:, -1:, :].squeeze()
        return self.activation(self.dense(cls_in).squeeze())

    def training_step(self, batch, batch_idx):
        x, x_time_enc, y, _ = batch
        y_hat = self(x, x_time_enc)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_time_enc, y, _ = batch
        y_hat = self(x, x_time_enc)
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
        x, x_time_enc, y, _ = batch
        y_hat = self(x, x_time_enc)
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
        optimizer = MADGRAD(self.parameters(), self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.lr_patience)
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
        parser.add_argument('--seq_len', type=int, default=256)
        parser.add_argument('--output_attention', type=bool, default=False)
        parser.add_argument('--enc_in', type=int, default=4)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--embed', type=str, default='timeF')
        parser.add_argument('--freq', type=str, default='d')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--e_layers', type=int, default=3)
        parser.add_argument('--factor', type=int, default=4)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--moving_avg', type=int, default=21)   # must be odd number.
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--lr_patience', type=int, default=6)
        return parser

# Autoformer of Trades model
class AoT(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        assert self.pred_len == 1, 'Pred len has to be 1!'
        self.autoformer = Autoformer.Model(configs)
        self.activation = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, y_dec_mark):
        dec_in = torch.zeros(
            x_enc.shape[0], self.pred_len, x_enc.shape[-1], dtype=torch.float32, device=x_enc.device)
        dec_in = torch.cat([x_enc[:, -self.label_len:, :], dec_in], dim=1)
        dec_mark_in = torch.cat([x_mark_enc[:, -self.label_len:, :], y_dec_mark.unsqueeze(1)], dim=1)
        output = self.autoformer(x_enc, x_mark_enc, dec_in, dec_mark_in)
        output = output[:, -1, -1].squeeze()
        return self.activation(output)

    def training_step(self, batch, batch_idx):
        x, x_time_enc, y, y_time_enc = batch
        y_hat = self(x, x_time_enc, y_time_enc)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_time_enc, y, y_time_enc = batch
        y_hat = self(x, x_time_enc, y_time_enc)
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
        x, x_time_enc, y, y_time_enc = batch
        y_hat = self(x, x_time_enc, y_time_enc)
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
        optimizer = MADGRAD(self.parameters(), self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.lr_patience)
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
        parser.add_argument('--seq_len', type=int, default=256)
        parser.add_argument('--output_attention', type=bool, default=False)
        parser.add_argument('--enc_in', type=int, default=4)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--embed', type=str, default='timeF')
        parser.add_argument('--freq', type=str, default='d')
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--e_layers', type=int, default=6)
        parser.add_argument('--factor', type=int, default=4)
        parser.add_argument('--n_heads', type=int, default=12)
        parser.add_argument('--d_ff', type=int, default=2048)
        parser.add_argument('--moving_avg', type=int, default=25)   # must be odd number.
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument('--learning_rate', type=float, default=0.00001)
        parser.add_argument('--lr_patience', type=int, default=6)
        parser.add_argument('--pred_len', type=int, default=1)
        parser.add_argument('--label_len', type=int, default=64)
        parser.add_argument('--dec_in', type=int, default=4)
        parser.add_argument('--d_layers', type=int, default=2)
        parser.add_argument('--c_out', type=int, default=4)
        return parser
