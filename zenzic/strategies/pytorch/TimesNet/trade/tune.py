from argparse import ArgumentParser
from zenzic.strategies.pytorch.TimesNet.trade.model import TimesNetTrades
from syne_tune.optimizer.baselines import HyperTune
from syne_tune.backend import LocalBackend
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint, uniform, finrange

import os
import logging

def main():
    logging.getLogger().setLevel(logging.INFO)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--syne_tune', type=bool, default=True)
    parser = TimesNetTrades.add_model_args(parser)
    args = parser.parse_args()

    config_space = {
        'max_epochs': args.max_epochs,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'syne_tune': args.syne_tune,
        'top_k': randint(lower=1, upper=5),
        'd_model': finrange(lower=4, upper=32, size=15, cast_int=True), # Even num in [4, 32]
        'd_ff': finrange(lower=4, upper=32, size=15, cast_int=True),    # Even num in [4, 32]
        'num_kernels': randint(lower=1, upper=10),
        'e_layers': randint(lower=1, upper=3),
        'dropout': uniform(lower=0.0, upper=1.0),
    }
    
    method_kwargs = dict(
        metric='val_loss',
        mode='min',
        resource_attr='epoch',
        max_resource_attr='max_epochs',
        grace_period=10,    # minimum 10 epochs for a trial
    )
    scheduler = HyperTune(
        config_space,
        search_options=None,
        type="promotion",
        brackets=1,
        **method_kwargs,
    )
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
    print(f"Training script is '{train_script}'.")
    backend = LocalBackend(entry_point=train_script, delete_checkpoints=True)
    stop_criterion = StoppingCriterion(
        max_wallclock_time=10*24*3600,   # 7 days
        max_num_trials_finished=200,
    )
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        sleep_time=0,
        # callbacks=[SimulatorCallback()],
        tuner_name="TimesNetTrades",
        # metadata={
        #     "seed": args.random_seed,
        #     "algorithm": args.method,
        #     "tag": args.experiment_tag,
        #     "benchmark": "nas201-" + args.dataset,
        # },
    )
    tuner.run()
    

if __name__ == '__main__':
    main()