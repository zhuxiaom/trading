set PYTHONPATH=%PYTHONPATH%;C:\Users\xzhu\Documents\GitHub\trading\
python -u run.py --is_training 1 --root_path C:\Trading\Autoformer\2022-01-14\ --watchlist SP500 --startdate 2010-01-01 --learning_rate 0.0002 --dropout 0.1 --model_id SP500 --model Autoformer --d_model 512 --n_heads 12 --data StockPrice --features MS --seq_len 256 --label_len 64 --pred_len 32 --e_layers 6 --d_layers 2 --factor 4 --enc_in 4 --dec_in 4 --c_out 4 --des 'SP500' --itr 1 --num_workers 0