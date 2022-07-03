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
        self.backbone = SCINet(
            output_len=configs.output_len,
            input_len=configs.window_size,
            input_dim=configs.input_dim,
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
        self.norm = nn.BatchNorm1d(configs.input_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1), bias=False)
        torch.nn.init.kaiming_uniform(self.conv.weight)
        self.dropout = nn.Dropout2d(p=configs.dropout)
        if self.backbone.output_len > 1:
            self.linear = nn.Linear(in_features=self.backbone.output_len, out_features=1, bias=False)
        else:
            self.linear = nn.Linear(in_features=self.backbone.input_dim, out_features=1, bias=False)
        torch.nn.init.kaiming_uniform(self.linear.weight)
        # self.linear.bias.data.fill_(0.01)
        self.activation = nn.Sigmoid()
        self.learning_rate = configs.learning_rate
        self.lr_patience = configs.lr_patience

    def forward(self, x):
        batch_norm_out = self.norm(torch.permute(x, (0, 2, 1)))
        backbone_out = self.backbone(torch.permute(batch_norm_out, (0, 2, 1)))
        if self.backbone.output_len > 1:
            conv_in = torch.unsqueeze(backbone_out, 1)
            conv_in = torch.permute(conv_in, (0, 1, 3, 2))
            conv_out = self.conv(conv_in)
            conv_out = self.dropout(conv_out)
            linear_out = self.linear(torch.squeeze(conv_out))
            return self.activation(torch.squeeze(linear_out))
        else:
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
        ### -------  input/output length settings --------------
        parser.add_argument('--window_size', type=int, default=64, help='input length')
        parser.add_argument('--input_dim', type=int, default=4, help='input length')
        parser.add_argument('--output_len', type=int, default=1, help='input length')
        parser.add_argument('--concat_len', type=int, default=165)
        parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
        ### -------  optimizer settings --------------
        parser.add_argument('--learning_rate', type=float,default=1e-3)
        parser.add_argument('--lr_patience', type=int, default=10)

        return parser