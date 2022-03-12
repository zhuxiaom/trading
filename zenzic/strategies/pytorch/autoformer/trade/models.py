from email.policy import default
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from zenzic.thirdparty.Autoformer.layers.Embed import DataEmbedding_wo_pos
from zenzic.thirdparty.Autoformer.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from zenzic.thirdparty.Autoformer.layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp
from madgrad import MADGRAD
from torchmetrics.functional import accuracy

# Autoformer in Trade model
class AiT(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.learning_rate = configs.learning_rate

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
            enc_out.shape[0], 1, enc_out.shape[2], device=('cuda:0' if enc_out.is_cuda else 'cpu'))
        enc_out = torch.cat((enc_out, cls), dim=1)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # output
        cls_in = enc_out[:, -1:, :].squeeze(dim=1)
        return self.activation(self.dense(cls_in))

    def training_step(self, batch, batch_idx):
        x, x_time_enc, y = batch
        y_hat = self(x, x_time_enc).squeeze(1)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, x_time_enc, y = batch
        y_hat = self(x, x_time_enc).squeeze(1)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, x_time_enc, y = batch
        y_hat = self(x, x_time_enc).squeeze(1)
        loss = F.binary_cross_entropy(y_hat, y.to(torch.float32))
        acc = accuracy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
         optimizer = MADGRAD(self.parameters(), self.learning_rate)
         return optimizer

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
        parser.add_argument('--n_heads', type=int, default=6)
        parser.add_argument('--d_ff', type=int, default=2048)
        parser.add_argument('--moving_avg', type=int, default=21)   # must be odd number.
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser
