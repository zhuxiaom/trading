from unicodedata import bidirectional
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from madgrad import MADGRAD
from torch.optim import Adam
from torchmetrics.functional import accuracy, f1_score, precision_recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from zenzic.thirdparty.SCINet.models.SCINet import SCINet
from argparse import ArgumentParser

# SCINet in Trades Classifier.
class SCINetTrades(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()

        # Input RNN layer.
        self.input_dim = configs.input_dim
        self.input_rnn_hidden_size = configs.input_rnn_hidden_size
        self.rnn = nn.GRU(
            input_size=configs.input_dim, hidden_size=self.input_rnn_hidden_size,
            num_layers=configs.input_rnn_layers, bias=False, batch_first=True,
            dropout=0.0, bidirectional=True)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        # Forward backbone network.
        self.fwd_backbone = SCINet(
            output_len=configs.output_len,
            input_len=configs.window_size,
            input_dim=self.input_rnn_hidden_size,
            hid_size=configs.hidden_size,
            num_stacks=configs.stacks,
            num_levels=configs.levels,
            num_decoder_layer=configs.num_decoder_layer,
            concat_len=configs.concat_len,
            groups=configs.groups,
            kernel=configs.kernel,
            dropout=configs.dropout,
            single_step_output_One=configs.single_step_output_One,
            positionalE=configs.positionalEcoding,
            modified=True,
            RIN=configs.RIN
        )
        # Backward backbone network.
        self.bwd_backbone = SCINet(
            output_len=configs.output_len,
            input_len=configs.window_size,
            input_dim=self.input_rnn_hidden_size,
            hid_size=configs.hidden_size,
            num_stacks=configs.stacks,
            num_levels=configs.levels,
            num_decoder_layer=configs.num_decoder_layer,
            concat_len=configs.concat_len,
            groups=configs.groups,
            kernel=configs.kernel,
            dropout=configs.dropout,
            single_step_output_One=configs.single_step_output_One,
            positionalE=configs.positionalEcoding,
            modified=True,
            RIN=configs.RIN
        )

        self.linear = nn.Linear(in_features=self.input_rnn_hidden_size * 2, out_features=1, bias=False)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        # self.linear.bias.data.fill_(0.01)
        self.activation = nn.Sigmoid()
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        rnn_out_fwd = rnn_out[:, :, 0:self.input_dim]
        rnn_out_bwd = rnn_out[:, :, self.input_dim:]

        fwd_backbone_out = self.fwd_backbone(rnn_out_fwd)
        bwd_backbone_out = self.bwd_backbone(rnn_out_bwd)
        backbone_out = torch.cat((fwd_backbone_out, bwd_backbone_out), dim=2)
        
        linear_out = self.linear(torch.squeeze(backbone_out))
        return self.activation(torch.squeeze(linear_out))
    
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
            cooldown=self.lr_patience, factor=0.5)
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
        parser.add_argument('--hidden_size', default=0.25, type=float, help='hidden channel of module')# H, EXPANSION RATE
        parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
        parser.add_argument('--kernel', default=5, type=int, help='kernel size')#k kernel size
        parser.add_argument('--dilation', default=1, type=int, help='dilation')
        parser.add_argument('--positionalEcoding', type=bool , default=False)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--groups', type=int, default=1)
        parser.add_argument('--levels', type=int, default=3)
        parser.add_argument('--num_decoder_layer', type=int, default=1)
        parser.add_argument('--stacks', type=int, default=1)
        parser.add_argument('--long_term_forecast', action='store_true', default=False)
        parser.add_argument('--RIN', type=bool, default=False)
        parser.add_argument('--input_rnn_hidden_size', type=int, default=4)
        parser.add_argument('--input_rnn_layers', type=int, default=2)
        ### -------  input/output length settings --------------
        parser.add_argument('--window_size', type=int, default=64, help='input length')
        parser.add_argument('--input_dim', type=int, default=4, help='input length')
        parser.add_argument('--output_len', type=int, default=1, help='output length')
        parser.add_argument('--concat_len', type=int, default=165)
        parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-3)
        parser.add_argument('--lr_patience', type=int, default=10)

        return parser