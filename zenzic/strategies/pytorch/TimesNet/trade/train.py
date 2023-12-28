from zenzic.strategies.pytorch.TimesNet.trade.model import TimesNetTrades
from zenzic.strategies.pytorch.data.trades import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging

import lightning.pytorch as pl
import os
import math

def main():
    pl.seed_everything(1688)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--downsamples', type=float, default=None)
    parser = TimesNetTrades.add_model_args(parser)
    args = parser.parse_args()
    # args.early_stopping = max(args.early_stopping, 2 * args.lr_patience + 1)

    # ------------
    # data
    # ------------
    train_data = Dataset(args.data_file, 'train', args.seq_len)
    val_data = Dataset(args.data_file, 'val', args.seq_len)
    if args.downsamples:
        print(f"Downsample ratio: {args.downsamples}")
        ds_offset = math.floor(len(train_data)*(1-args.downsamples))
        train_data = train_data.to_gpu(offset=-ds_offset)
        ds_offset = math.floor(len(val_data)*args.downsamples)
        val_data = val_data.to_gpu(offset=ds_offset)
    else:
        train_data = train_data.to_gpu()
        val_data = val_data.to_gpu()
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
    logger = TensorBoardLogger(
        save_dir=args.output_dir, name=model.__class__.__name__)
    logger.log_hyperparams(args)
    lr_logger = LearningRateMonitor()  # log the learning rate
    save_model = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(logger.log_dir, 'best_models'),
        filename="TimesNet-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=False,
    )
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start=6, swa_lrs=0.01, device=None)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=args.early_stopping,
        verbose=False,
        mode="min")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        enable_model_summary=True,
        callbacks=[lr_logger, save_model, early_stop_callback],
        logger=logger,
        default_root_dir=args.output_dir)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=test_loader)
    # print(result)

if __name__ == '__main__':
    main()