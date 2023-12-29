
import argparse
import os
import pickle

from zenzic.strategies.pytorch.data.trades import Dataset

def save_dataset(args, type):
    x, date_x, y, date_y = Dataset(args.in_file, type, args.seq_len).fetch_all()
    data = dict(
        x = x,
        date_x = date_x,
        y = y,
        date_y = date_y
    )
    path = os.path.join(args.out_dir, type + '.pkl')
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data samples to train, validation and test.')
    parser.add_argument('--in_file', type=str, required=True, default=None, help='The file of WealthLab trades.')
    parser.add_argument('--out_dir', type=str, required=True, default=None, help='The output directory.')
    parser.add_argument('--seq_len', type=int, required=False, default=256, help='The max length of quotes.')
    args = parser.parse_args()

    save_dataset(args, 'train')
    save_dataset(args, 'val')
    save_dataset(args, 'test')
