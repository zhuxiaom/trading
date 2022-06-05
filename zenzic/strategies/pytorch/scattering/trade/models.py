import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import kymatio.numpy as knp
import numpy
import vit_pytorch

from argparse import ArgumentParser
from madgrad import MADGRAD
from torchmetrics.functional import accuracy, f1_score, precision_recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from kymatio.torch import Scattering1D

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    i = 0
    j = 1
    while j < len(primfac):
        if (primfac[i] != primfac[j]):
            i = i + 1
            primfac[i] = primfac[j]
        j = j + 1
    return primfac[:(i+1)]

# Wavelet Scattering + SimpleViT model.
class SViT(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.num_of_signals = configs.num_of_signals
        self.time_scale = configs.time_scale
        self.spatial_scale = configs.spatial_scale
        self.vit_dim = configs.vit_dim
        self.vit_depth = configs.vit_depth
        self.vit_heads = configs.vit_heads
        self.vit_mlp_dim = configs.vit_mlp_dim
        self.vit_dim_head = configs.vit_dim_head
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

        # Wavelet Scattering Layer
        ws_shape = None
        try:
            # Sanity check if wavelet scattering will fail with the config.
            t = numpy.random.rand(configs.seq_len)
            ws_shape = knp.Scattering1D(self.spatial_scale, self.seq_len, self.time_scale)(t).shape
        except:
            raise
        self.scattering = Scattering1D(self.spatial_scale, self.seq_len, self.time_scale)
        patch_size = (primes(ws_shape[0])[0], primes(ws_shape[1])[0])
        self.vit = vit_pytorch.SimpleViT(
            image_size=ws_shape, patch_size=patch_size, num_classes=1, dim=self.vit_dim,
            depth=self.vit_depth, heads=self.vit_heads, mlp_dim=self.vit_mlp_dim,
            channels=self.num_of_signals, dim_head=self.vit_dim_head)
        self.activation = nn.Sigmoid()

    def forward(self, x_enc, x_mark_enc, enc_self_mask=None):
        scattering_out = self.scattering(x_enc)
        cls_out = self.vit(scattering_out)
        return self.activation(cls_out.squeeze())

    def training_step(self, batch, batch_idx):
        x, x_mark_enc, y, _ = batch
        y_hat = self(x, x_mark_enc)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_mark_enc, y, _ = batch
        y_hat = self(x, x_mark_enc)
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
        x, x_mark_enc, y, _ = batch
        y_hat = self(x, x_mark_enc)
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
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=self.lr_patience)
        if self.lr_patience <= 0:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_f1"
            }

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--seq_len', type=int, default=256)
        parser.add_argument('--num_of_signals', type=int, default=4)
        parser.add_argument('--time_scale', type=int, default=8)
        parser.add_argument('--spatial_scale', type=int, default=5)
        parser.add_argument('--vit_dim', type=int, default=1024)
        parser.add_argument('--vit_depth', type=int, default=8)
        parser.add_argument('--vit_heads', type=int, default=8)
        parser.add_argument('--vit_mlp_dim', type=int, default=2048)
        parser.add_argument('--vit_dim_head', type=int, default=64)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--lr_patience', type=int, default=10)
        return parser
