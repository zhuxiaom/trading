from argparse import ArgumentParser
from zenzic.strategies.pytorch.TimesNet.trade.model import TimesNetTrades
from syne_tune.optimizer.baselines import HyperTune
from syne_tune.backend import LocalBackend
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import randint, uniform, finrange
from syne_tune.tuner_callback import TunerCallback
from syne_tune.util import RegularCallback
from syne_tune.backend.trial_status import Status
from collections import OrderedDict
from syne_tune.results_callback import StoreResultsCallback
from datetime import timedelta

import os
import logging
import pandas as pd
import time

logger = logging.getLogger(__name__)

def tunig_status_to_string(tuning_status):
    blacklist_cols = {'input_dir', 'output_dir', 'lr_patience'}
    num_running = tuning_status.num_trials_running
    num_finished = tuning_status.num_trials_started - num_running

    if len(tuning_status.trial_rows) > 0:
        running_trials = OrderedDict({})
        for trial_id, row in tuning_status.trial_rows.items():
            if row["status"] == Status.in_progress:
                running_trials[trial_id] = row
        df = pd.DataFrame(running_trials.values())
        cols = [col for col in df.columns if not col.startswith("st_") and col not in blacklist_cols]
        res_str = df.loc[:, cols].to_string(index=False, na_rep="-") + "\n"
    else:
        res_str = ""
    res_str += (
        f"{num_running} trials running, "
        f"{num_finished} finished ({tuning_status.num_trials_completed} until the end), "
        f"{timedelta(seconds=tuning_status.wallclock_time)} elapsed."
    )
    # f"{self.user_time:.2f}s approximated user-time"
    cost = tuning_status.cost
    if cost is not None and cost > 0.0:
        res_str += f", ${cost:.2f} estimated cost"
    res_str += "\n"
    return res_str

class SyneTuneStatus(TunerCallback):
    def on_tuning_start(self, tuner):
        tuner.status_printer = RegularCallback(
            callback=lambda tuning_status: logger.info(
                "tuning status (last metric is reported)\n" + tunig_status_to_string(tuning_status)
            ),
            call_seconds_frequency=tuner.print_update_interval,
        )

def main():
    logging.getLogger().setLevel(logging.INFO)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser = TimesNetTrades.add_model_args(parser)
    args = parser.parse_args()

    config_space = {
        'max_epochs': args.max_epochs,
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'lr_patience': 0,
        'norm_mode': 1,
        'seq_len': randint(lower=128, upper=256),
        'top_k': 1,
        'd_model': finrange(lower=4, upper=64, size=31, cast_int=True), # Even num in [4, 64]
        'd_ff': finrange(lower=4, upper=64, size=31, cast_int=True), # Even num in [4, 64]
        'num_kernels': finrange(lower=4, upper=10, size=4, cast_int=True), # Even num in [4, 10]
        'e_layers': finrange(lower=4, upper=10, size=4, cast_int=True), # Even num in [4, 10]
        'dropout': uniform(lower=0.0, upper=0.5),
    }
    
    method_kwargs = dict(
        metric='val_loss',
        mode='min',
        resource_attr='epoch',
        max_resource_attr='max_epochs',
        grace_period=10,    # minimum 10 epochs for a trial
        random_seed=int(time.time()),
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
        max_wallclock_time=7*24*3600,   # 7 days
        max_num_trials_finished=200,
    )
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=1,
        sleep_time=0,
        callbacks=[StoreResultsCallback(), SyneTuneStatus()],
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