from typing import Any
from zenzic.strategies.pytorch.PatchTST.trade.model import PatchTSTTrades
from zenzic.strategies.pytorch.data.trades import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from torch.utils.data import TensorDataset
from lightning.pytorch.callbacks import Callback
from dehb import DEHB

import lightning.pytorch as pl
import os
import pickle
import torch
import time
import numpy as np
import math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class ValLoss(Callback):
    def __init__(self) -> None:
        self.value = 1.0

    def on_validation_epoch_end(self, trainer, pl_module):
        self.value = min(self.value, trainer.logged_metrics['val_loss'])
    
class Experiment:
    def __init__(self, config) -> None:
        self.__hp_params = config

        self.__train_data = Dataset(config.data_file, 'train', config.seq_len)
        self.__val_data = Dataset(config.data_file, 'val', config.seq_len)
        if config.downsamples:
            print(f"Downsample ratio: {self.__hp_params.downsamples}")
            ds_offset = math.floor(len(self.__train_data)*(1-self.__hp_params.downsamples))
            self.__train_data = self.__train_data.fetch_all(offset=-ds_offset)
            ds_offset = math.floor(len(self.__val_data)*self.__hp_params.downsamples)
            self.__val_data = self.__val_data.fetch_all(offset=ds_offset)
        else:
            self.__train_data = self.__train_data.fetch_all()
            self.__val_data = self.__val_data.fetch_all()
        
        # Init config space
        self.__config_space = CS.ConfigurationSpace()
        attn_dropout = CSH.UniformFloatHyperparameter('attn_dropout', lower=0.0, upper=1.0, default_value=0.0)
        dropout = CSH.UniformFloatHyperparameter('dropout', lower=0.0, upper=1.0, default_value=0.0)
        head_dropout = CSH.UniformFloatHyperparameter('head_dropout', lower=0.0, upper=1.0, default_value=0.0)
        patch_len = CSH.UniformIntegerHyperparameter('patch_len', lower=1, upper=128, default_value=128)
        e_layers = CSH.UniformIntegerHyperparameter('e_layers', lower=1, upper=3, default_value=1)
        d_ff = CSH.UniformIntegerHyperparameter('d_ff', lower=1, upper=256, default_value=256)
        pred_len = CSH.UniformIntegerHyperparameter('pred_len', lower=1, upper=256, default_value=20)
        n_heads = CSH.UniformIntegerHyperparameter('n_heads', lower=1, upper=16, default_value=8)
        d_model_factor = CSH.UniformIntegerHyperparameter('n_model_factor', lower=1, upper=16, default_value=16)
        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=1, upper=128, default_value=128)
        self.__config_space.add_hyperparameters([attn_dropout, dropout, head_dropout, patch_len, e_layers,
                                                 d_ff, pred_len, n_heads, d_model_factor, kernel_size,])
        
    def get_config_space(self):
        return self.__config_space

    def __call__(self, config, budget) -> Any:
        dst = torch.device(device=self.__hp_params.device)
        train_dataset = TensorDataset(
            torch.tensor(np.asanyarray(self.__train_data[0]), dtype=torch.float32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__train_data[1]), dtype=torch.float32,device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__train_data[2]), dtype=torch.int32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__train_data[3]), dtype=torch.float32, device=dst).share_memory_())
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.__hp_params.batch_size,
            num_workers=0,
            shuffle=True,
            persistent_workers=False)
        val_dataset = TensorDataset(
            torch.tensor(np.asanyarray(self.__val_data[0]), dtype=torch.float32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__val_data[1]), dtype=torch.float32,device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__val_data[2]), dtype=torch.int32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(self.__val_data[3]), dtype=torch.float32, device=dst).share_memory_())
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.__hp_params.batch_size,
            num_workers=0,
            persistent_workers=False)
        
        # ------------
        # hyper-parameters
        # ------------
        self.__hp_params.attn_dropout = config['attn_dropout']
        self.__hp_params.dropout = config['dropout']
        self.__hp_params.head_dropout = config['head_dropout']
        self.__hp_params.patch_len = config['patch_len']
        self.__hp_params.e_layers = config['e_layers']
        self.__hp_params.d_ff = config['d_ff']
        self.__hp_params.pred_len = config['pred_len']
        self.__hp_params.n_heads = config['n_heads']
        self.__hp_params.d_model = config['n_model_factor'] * self.__hp_params.n_heads
        self.__hp_params.kernel_size = config['kernel_size']

        # ------------
        # training
        # ------------
        start = time.time()  # measuring wallclock time
        model = PatchTSTTrades(self.__hp_params)
        logger = TensorBoardLogger(
            save_dir=self.__hp_params.output_dir, name=model.__class__.__name__)
        logger.log_hyperparams(self.__hp_params, metrics={"min_loss": 0})
        lr_logger = LearningRateMonitor()  # log the learning rate
        val_loss = ValLoss()
        # save_model = ModelCheckpoint(
        #     monitor="val_loss",
        #     dirpath=os.path.join(logger.log_dir, 'best_models'),
        #     filename="PatchTST-{epoch:03d}-{val_loss:.6f}",
        #     save_top_k=3,
        #     mode="min",
        #     save_on_train_epoch_end=False,
        # )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=self.__hp_params.early_stopping,
            verbose=False,
            mode="min")
        trainer = pl.Trainer(
            max_epochs=round(budget),
            accelerator="gpu",
            enable_model_summary=True,
            callbacks=[lr_logger, early_stop_callback, val_loss],
            logger=logger,
            default_root_dir=self.__hp_params.output_dir,
            num_sanity_val_steps=0)
        trainer.fit(model, train_loader, val_loader)
        cost = time.time() - start
        logger.log_metrics(metrics={'min_loss': val_loss.value})

        # dict representation that DEHB requires
        res = {
            "fitness": val_loss.value,
            "cost": cost,
            "info": {"budget": budget}
        }
        return res

def main():
    # pl.seed_everything(1688)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--downsamples', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser = PatchTSTTrades.add_model_args(parser)
    args = parser.parse_args()
    # args.early_stopping = max(args.early_stopping, 2 * args.lr_patience + 1)

    exp = Experiment(args)
    config_space = exp.get_config_space()
    dimensions = len(config_space.get_hyperparameters())
    dehb = DEHB(f=exp, cs=config_space, dimensions=dimensions, min_budget=10,
                max_budget=50, eta=3, output_path=args.output_dir,
                # if client is not None and of type Client, n_workers is ignored
                # if client is None, a Dask client with n_workers is set up
                client=None, n_workers=1)
    _, _, history = dehb.run(fevals=200, verbose=True,
                             # arguments below are part of **kwargs shared across workers
                             single_node_with_gpus=True)
    name = time.strftime("%x %X %Z", time.localtime(dehb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    dehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_dir, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main()