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

import lightning.pytorch as pl
import torch
import numpy as np
import math
import optuna

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

    def __call__(self, trial) -> Any:
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
        self.__hp_params.attn_dropout = trial.suggest_float('attn_dropout', 0.0, 1.0)
        self.__hp_params.dropout = trial.suggest_float('dropout', 0.0, 1.0)
        self.__hp_params.head_dropout = trial.suggest_float('head_dropout', 0.0, 1.0)
        self.__hp_params.patch_len = trial.suggest_int('patch_len', 1, 128)
        self.__hp_params.e_layers = trial.suggest_int('e_layers', 1, 3)
        self.__hp_params.d_ff = trial.suggest_int('d_ff', 1, 256)
        self.__hp_params.pred_len = trial.suggest_int('pred_len', 1, 256)
        self.__hp_params.n_heads = trial.suggest_int('n_heads', 1, 16)
        d_model_factor = trial.suggest_int('n_model_factor', 1, 16)
        self.__hp_params.d_model = self.__hp_params.n_heads * d_model_factor
        self.__hp_params.kernel_size = trial.suggest_int('kernel_size', 1, 128)

        # ------------
        # training
        # ------------
        model = PatchTSTTrades(self.__hp_params)
        logger = TensorBoardLogger(
            save_dir=self.__hp_params.output_dir, name=model.__class__.__name__)
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
            min_delta=1e-3,
            patience=self.__hp_params.early_stopping,
            verbose=False,
            mode="min")
        trainer = pl.Trainer(
            max_epochs=self.__hp_params.max_epochs,
            accelerator="gpu",
            enable_model_summary=True,
            callbacks=[lr_logger, early_stop_callback, val_loss],
            logger=logger,
            default_root_dir=self.__hp_params.output_dir,
            num_sanity_val_steps=0)
        trainer.fit(model, train_loader, val_loader)
        logger.log_hyperparams(self.__hp_params, metrics={'val_loss': val_loss.value})
        return val_loss.value

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

    # ------------
    # hyper-parameter tuning
    # ------------
    # sampler = optuna.samplers.CmaEsSampler(use_separable_cma=True)
    sampler = optuna.samplers.TPESampler(multivariate=True) 
    study_name = "Tune-PatchTST"
    storage_name = "sqlite:///" + args.output_dir + "/optuna.db"
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_name, load_if_exists=True)
    exp = Experiment(args)
    study.optimize(lambda t: exp(t), n_trials=200)
    print("======== Best Params ========")
    print(study.best_params)
    print("======== Best Trial ========")
    print(study.best_trial)
    print("======== Best Value ========")
    print(study.best_value)

if __name__ == '__main__':
    main()