import os
import zenzic.strategies.pytorch.data.stock_price as stock_price
import zenzic.strategies.pytorch.data.oxford_volatility as oxford_volatility

from torch.utils.data import DataLoader

data_dict = {
    'StockPrice': stock_price.Dataset,
    'OxfordVolatility': oxford_volatility.Dataset,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_Pred
        raise NotImplementedError("Not implemented!")
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = None
    if args.data == 'StockPrice':
        data_set = Data(
            watchlist=args.watchlist,
            startdate=args.startdate,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            # timeenc=timeenc,
            # freq=freq
        )
    elif  args.data == 'OxfordVolatility':
         data_set = Data(
            data_file=os.path.join(args.root_path, args.data_file),
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
