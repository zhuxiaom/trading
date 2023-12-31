from zenzic.strategies.pytorch.TimesNet.trade.model import TimesNetTrades
from zenzic.strategies.pytorch.data.trades import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from torch.utils.data import TensorDataset
from syne_tune.report import Reporter
from lightning.pytorch.callbacks import Callback
from syne_tune.constants import ST_CHECKPOINT_DIR

import lightning.pytorch as pl
import os
import pickle
import torch
import numpy as np

class SyneTuneReporter(Callback):
    def __init__(self, config) -> None:
        self.config = config
        self.epoch = 0
        self.losses = []
        self.min_loss = float('inf')
        if self.config.syne_tune:
            self.reporter = Reporter(add_time=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.epoch += 1
        self.losses.append(trainer.logged_metrics['val_loss'].item())
        self.min_loss = min(self.min_loss, self.losses[-1])
        # Delay reporting last value to allow the script to complete.
        if self.epoch < self.config.max_epochs:
            self.report()
    
    def report(self):
        if self.config.syne_tune:
            self.reporter(epoch=self.epoch, val_loss=self.losses[-1])

    def finish(self):
        if self.epoch == self.config.max_epochs:
            self.report()

def load_dataset(dir, type, device):
    path = os.path.join(dir, type + '.pkl')
    print(f"Load {type} dataset from '{path}'.")
    with open(path, 'rb') as file:
        data = pickle.load(file)
        x, date_x, y, date_y = data['x'], data['date_x'], data['y'], data['date_y']
        dst = torch.device(device=device)
        return TensorDataset(
            torch.tensor(np.asanyarray(x), dtype=torch.float32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(date_x), dtype=torch.float32,device=dst).share_memory_(),
            torch.tensor(np.asanyarray(y), dtype=torch.int32, device=dst).share_memory_(),
            torch.tensor(np.asanyarray(date_y), dtype=torch.float32, device=dst).share_memory_())


def main():
    pl.seed_everything(1688)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--syne_tune', type=bool, default=False)
    # Syne Tune args
    parser.add_argument(f'--{ST_CHECKPOINT_DIR}', type=str, default="./")
    parser = TimesNetTrades.add_model_args(parser)
    args = parser.parse_args()
    # args.early_stopping = max(args.early_stopping, 2 * args.lr_patience + 1)

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
    model = TimesNetTrades(args)
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir, name=model.__class__.__name__, default_hp_metric=False)
    tb_logger.log_hyperparams(args, metrics={'min_loss': 0})
    lr_logger = LearningRateMonitor()  # log the learning rate
    st_reporter = SyneTuneReporter(config=args)
    save_model = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(tb_logger.log_dir, 'best_models'),
        filename="TimesNet-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=False,
    )
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start=6, swa_lrs=0.01, device=None)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-6,
        patience=args.early_stopping,
        verbose=False,
        mode="min")
    callbacks = [lr_logger, early_stop_callback, st_reporter]
    if not args.syne_tune:
        callbacks.append(save_model)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        enable_model_summary=True,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=args.output_dir,
        enable_checkpointing=not args.syne_tune,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.syne_tune,
    )
    trainer.fit(model, train_loader, val_loader)
    tb_logger.log_metrics(metrics={'min_loss': st_reporter.min_loss})

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)
	
    st_reporter.finish()
	

if __name__ == '__main__':
    main()