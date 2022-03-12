import torch
from zenzic.strategies.pytorch.autoformer.trade.models import AiT
from zenzic.strategies.pytorch.data.trades import Dataset
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

import pytorch_lightning as pl
import os

def main():
    pl.seed_everything(1688)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser = AiT.add_model_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_loader = DataLoader(
        Dataset(args.data_file, 'train', args.seq_len), batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(
        Dataset(args.data_file, 'val', args.seq_len), batch_size=args.batch_size, num_workers=2)
    # test_loader = DataLoader(
    #     Dataset(args.data_file, 'test', args.seq_len), batch_size=args.batch_size, num_workers=3)

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger(
        save_dir=args.output_dir, name='tensorboard_logs')
    logger.log_hyperparams(args)
    model = AiT(args)
    lr_logger = LearningRateMonitor()  # log the learning rate
    save_model = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(logger.log_dir, 'best_models'),
        filename="ait-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=False,
    )
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start=6, swa_lrs=0.01, device=None)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        weights_summary="top",
        callbacks=[lr_logger, save_model, stoch_weight_avg],
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