import os
import argparse
import pandas as pd
import math
import warnings
import torch
import pytorch_lightning as pl
import zenzic.strategies.pytorch.tft.data_processor as dp

from ast import arg
from datetime import datetime
from email.policy import default
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_lightning.callbacks import ModelCheckpoint
from madgrad import MADGRAD

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Fusion Transformer model predicts stock price.')
    parser.add_argument('--root_dir', type=str, required=True, default=None, help='The root path of the data file.')
    parser.add_argument('--encoder_len', type=int, default=256,  help='The encoder length.')
    parser.add_argument('--pred_len', type=int, default=32,  help='The length of prediction.')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='The start date of historical stock prices used for training.')
    parser.add_argument('--end_date', type=str, default=None, help='The end date of historical stock prices used for training.')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--dropout', type=float, default=0.3, help='The dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=30, help='The max num of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate.')
    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    prices = dp.sp500_prices(start_date=args.start_date, end_date=args.end_date)
    date_features = dp.date_features(prices)
    prices = prices.join(date_features, on='Date', how='left')

    df_train, df_pred = dp.split_data(prices, args.encoder_len, args.pred_len, 0.7)
    df_val, df_test = dp.split_data(df_pred, args.encoder_len, args.pred_len, 1.0 / 3)

    train_data = TimeSeriesDataSet(
        df_train,
        time_idx='Date_idx',
        target='Close',
        group_ids=['Symbol'],
        max_encoder_length=args.encoder_len,
        max_prediction_length=args.pred_len,
        time_varying_known_reals=[
            'Date_idx',
            'Day_of_year',
            'Day_of_week',
            'Month',
            'Day'],
        time_varying_unknown_reals=[
            'Open',
            'High',
            'Low',
            'Close'
        ],
        target_normalizer=TorchNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=False,
    )
    val_data = TimeSeriesDataSet.from_dataset(train_data, df_val, predict=False, stop_randomization=True)
    test_data = TimeSeriesDataSet.from_dataset(train_data, df_test, predict=False, stop_randomization=True)
    print(f'Train data: {len(train_data)}, Val data: {len(val_data)}, Test data: {len(test_data)}')
    train_dataloader = train_data.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_dataloader = val_data.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    test_dataloader = test_data.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=6,
        verbose=False,
        mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(
        os.path.join(args.root_dir, 'tensorboard_logs'))  # logging results to a tensorboard
    save_model = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(args.root_dir, 'best_models'),
        filename="tft-{epoch:03d}-{val_loss:.6f}",
        save_top_k=3,
        mode="min",
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=1,
        weights_summary="top",
        callbacks=[lr_logger, save_model, early_stop_callback],
        logger=logger,
        default_root_dir=args.root_dir,
        # fast_dev_run=True,
    )

    tft = TemporalFusionTransformer.from_dataset(
        train_data,
        learning_rate=args.learning_rate,
        lstm_layers=2,
        hidden_size=384,
        attention_head_size=4,
        dropout=args.dropout,
        hidden_continuous_size=2048,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # log_interval=1.0,
        reduce_on_plateau_patience=4,
        optimizer=MADGRAD
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
