from zenzic.strategies.pytorch.PatchMixer.price.model import PatchMixerPrice
from zenzic.strategies.pytorch.data.trades import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from torch.utils.data import TensorDataset
from syne_tune.report import Reporter
from lightning.pytorch.callbacks import Callback
from syne_tune.constants import ST_CHECKPOINT_DIR, ST_WORKER_ITER, ST_WORKER_TIME

import lightning.pytorch as pl
import os
import pickle
import torch
import numpy as np
import ast
import time
import random

__CHECKPOINT_FILE__ = 'last.chkp'
__RNG_STATES_FILE__ = 'rng_states.chkp'

class SyneTuneReporter(Callback):
    def __init__(self, trial_root: str) -> None:
        self.maes = []
        self.cum_maes = []
        self.min_mae = float('inf')
        self.min_cum_mae = float('inf')
        self.losses = []
        self.min_loss = float('inf')
        self.reporter = Reporter(add_time=True)
        self.trial_root = trial_root
        self.score = float('inf')
        if self.trial_root is not None:
            # Try to recover from checkpoint.
            std_out = os.path.join(self.trial_root, 'std.out')
            if os.path.isfile(std_out):
                with open(std_out, 'r') as file:
                    max_worker_time = 0
                    for line in file:
                        if line.startswith('[tune-metric]:'):
                            metrics_line = line.replace('[tune-metric]:', '').strip()
                            metrics = ast.literal_eval(metrics_line)
                            self.reporter.iter = metrics[ST_WORKER_ITER] + 1
                            max_worker_time = max(max_worker_time, metrics[ST_WORKER_TIME])
                            self.losses.append(metrics['val_loss'])
                            self.maes.append(metrics['val_mae'])
                            self.cum_maes.append(metrics['val_cum_mae'])
                            self.min_loss = min(self.min_loss, self.losses[-1])
                    self.reporter.start -= max_worker_time

    def on_validation_epoch_end(self, trainer, pl_module):
        self.maes.append(trainer.logged_metrics['val_mae'].item())
        self.min_mae = min(self.maes)
        self.cum_maes.append(trainer.logged_metrics['val_cum_mae'].item())
        self.min_cum_mae = min(self.cum_maes)
        self.losses.append(trainer.logged_metrics['val_loss'].item())
        self.min_loss = min(self.losses)
        losses = np.array(self.losses)
        min_idx = np.argmin(losses)
        self.score = (np.mean(losses[min_idx:]) / losses[min_idx] - 1) * 100
        epoch = trainer.current_epoch + 1   # current_epoch is updated after validation ends.
        # Delay reporting last value to allow the script to complete.
        if self.trial_root is not None and epoch < trainer.max_epochs:
            self.report(epoch)
    
    def report(self, epoch):
        self.reporter(epoch=epoch,
                      score=self.score,
                      val_loss=self.losses[-1],
                      val_mae=self.maes[-1],
                      val_cum_mae=self.cum_maes[-1],
                      )

    def finish(self, trainer):
        if self.trial_root is not None and trainer.current_epoch == trainer.max_epochs:
            self.report(trainer.current_epoch)

def load_dataset(dir, type, device):
    path = os.path.join(dir, type + '.pkl')
    print(f"Load {type} dataset from '{path}'.")
    with open(path, 'rb') as file:
        data = pickle.load(file)
        x, date_x, y, date_y = data['x'], data['date_x'], data['y'], data['date_y']
        dst = torch.device(device=device)
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32, device=dst).share_memory_(),
            torch.tensor(date_x, dtype=torch.float32,device=dst).share_memory_(),
            torch.tensor(y, dtype=torch.float32, device=dst).share_memory_(),
            torch.tensor(date_y, dtype=torch.float32, device=dst).share_memory_())

def find_st_trial(st_checkpoit_dir: str):
    # e.g. C:\Trading\PatchMixerPrice\2023-12-27\PatchMixerPrice-2024-01-01-18-17-39-789\0\checkpoints
    head, tail = os.path.split(st_checkpoit_dir)
    if tail == "checkpoints":
        head, tail = os.path.split(head)
        return head, tail
    else:
        return find_st_trial(head)

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--rand_seed', type=int, default=1688)
    parser.add_argument('--matmul_prec', type=str, default='high')
    # Syne Tune args
    parser.add_argument(f'--{ST_CHECKPOINT_DIR}', type=str, default=None)
    parser = PatchMixerPrice.add_model_args(parser)
    args = parser.parse_args()
    # args.early_stopping = max(args.early_stopping, 2 * args.lr_patience + 1)
    torch.set_float32_matmul_precision(args.matmul_prec)

    checkpoint_dir = getattr(args, ST_CHECKPOINT_DIR)
    syne_tune_enabled = checkpoint_dir is not None
    checkpoint_file = (os.path.join(checkpoint_dir, __CHECKPOINT_FILE__) if syne_tune_enabled else None)
    rng_states_file = (os.path.join(checkpoint_dir, __RNG_STATES_FILE__) if syne_tune_enabled else None)
    checkpoint_exists = (os.path.isfile(checkpoint_file) if syne_tune_enabled else False)
    trial_root, trial_id = None, None
    if syne_tune_enabled:
        trial_root, trial_id = find_st_trial(checkpoint_dir)
        args.output_dir = trial_root

    # ------------
    # data
    # ------------
    train_data = load_dataset(dir=args.input_dir, type='train', device='cuda:0')
    val_data = load_dataset(args.input_dir, type='val', device='cuda:0')
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        persistent_workers=False)
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=0,
        persistent_workers=False)
    # test_loader = DataLoader(
    #     Dataset(args.data_file, 'test', args.seq_len), batch_size=args.batch_size, num_workers=3)

    # ------------
    # training
    # ------------
    pl.seed_everything(args.rand_seed)
    if checkpoint_exists:
        model = PatchMixerPrice.load_from_checkpoint(checkpoint_file, configs=args)
    else:
        model = PatchMixerPrice(args)
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=("" if syne_tune_enabled else model.__class__.__name__),
        version=trial_id,
        default_hp_metric=False,
    )
    tb_logger.log_hyperparams(args, metrics={'min_loss': 0,
                                             'min_mae': 0,
                                             'min_cum_mae': 0,
                                             'score': 0})
    lr_logger = LearningRateMonitor()  # log the learning rate
    st_reporter = SyneTuneReporter(
        (os.path.join(trial_root, trial_id) if syne_tune_enabled else None))
    save_model = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(tb_logger.log_dir, 'best_models'),
        filename="PatchMixerPrice-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=False,
    )
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start=6, swa_lrs=0.01, device=None)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=args.learning_rate/1000,
        patience=args.early_stopping,
        verbose=False,
        mode="min")
    callbacks = [lr_logger, early_stop_callback, st_reporter]
    if not syne_tune_enabled:
        callbacks.append(save_model)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        enable_model_summary=True,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=args.output_dir,
        enable_checkpointing=not syne_tune_enabled,
        num_sanity_val_steps=0,
        enable_progress_bar=not syne_tune_enabled,
        # detect_anomaly=True,
    )
    if not checkpoint_exists:
        trainer.fit(model, train_loader, val_loader)
    else:
        rng_states = torch.load(rng_states_file)
        torch.set_rng_state(rng_states['cpu_rng_state'])
        torch.cuda.set_rng_state(rng_states['gpu_rng_state'])
        np.random.set_state(rng_states['numpy_rng_state'])
        random.setstate(rng_states['py_rng_state'])
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_file)
    tb_logger.log_metrics(metrics={'min_loss': st_reporter.min_loss,
                                   'min_mae': st_reporter.min_mae,
                                   'min_cum_mae': st_reporter.min_cum_mae,
                                   'score': st_reporter.score})
    if syne_tune_enabled:
        trainer.save_checkpoint(checkpoint_file)
        rng_states = {
            'cpu_rng_state': torch.get_rng_state(),
            'gpu_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'py_rng_state': random.getstate()
        }
        torch.save(rng_states, rng_states_file)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)
	
    if syne_tune_enabled:
        st_reporter.finish(trainer)

if __name__ == '__main__':
    main()