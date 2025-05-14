import numpy as np
import pandas as pd
import torch
from ray import tune
from concurrent.futures import ThreadPoolExecutor
import quantstats as qs
import statistics

from Components.TrainModel import torchscript_predict
from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

# ─── 1. Load the data ────────────────────────────────────

_INDEX_CONFIG = {
    'NASDAQ':        ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 1),
    'S&P500':        ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 0),
    'RUSSELL1000':   ('https://en.wikipedia.org/wiki/Russell_1000_Index', 3, 1),
    'DOWJONES':      ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 2, 2),
}

def get_index_tickers(indices, sample_size=10):
    all_tickers = []
    for idx in indices:
        cfg = _INDEX_CONFIG.get(idx)
        if not cfg:
            continue
        url, table_i, col_i = cfg
        try:
            df = pd.read_html(url)[table_i]
            tickers = df.iloc[:, col_i].dropna().astype(str).tolist()
        except Exception as e:
            print(f"Warning: could not fetch {idx} → {e}")
            continue
        all_tickers.extend(tickers)
    all_tickers = list(dict.fromkeys(all_tickers))
    if sample_size >= len(all_tickers):
        raise ValueError(f"Sample size ({sample_size}), cannot be greater than length of list ({len(all_tickers)}) !")
    else:
        sampled_tickers = random.sample(list(all_tickers), sample_size)

    return sampled_tickers

indicators = ['ema_20', 'ema_50', 'ema_100', 'stoch_rsi14', 'macd', 'State', 'Close']
tickers = ['IONQ','AAPL']
out_of_sample_data, raw_stock_data = TickerData(tickers, indicators, years=1, prediction_window=5, prediction_mode=True).process_all()
if out_of_sample_data is None:
    raise ValueError("No data retrieved!")
out_of_sample_data = out_of_sample_data[['Ticker','ema_20', 'ema_50', 'ema_100', 'stoch_rsi14', 'macd', 'State', 'Close']]
index_returns = TickerData('I:NDX', [], years=1).get_ohlc_for_ticker('I:NDX')['close'].pct_change().tz_localize(None)
if index_returns is None:
    raise ValueError("No index data retrieved!")

# ─── 2. Utility functions ────────────────────────────────────────────────────

def predict_backtest(ticker, pred_data, raw_data, index_returns, pct_change_entry, pct_change_exit, model_window_size):
    pred_data_t = pred_data[pred_data['Ticker']==ticker].copy()
    preds = torchscript_predict(
        model_path="Models/Tempus_v2.pt",
        input_df=pred_data_t,
        device="cpu",
        window_size=model_window_size,
        prediction_mode=True
    )
    raw_data_t = raw_data[raw_data['Ticker']==ticker].copy()

    final_pred_data = pd.merge(preds, raw_data_t[['open', 'high', 'low', 'volume','close']], left_index=True, right_index=True, how='left')
    
    backtester = BackTesting(final_pred_data.rename(columns={'close': 'Close'}), ticker, 1000, pct_change_entry, pct_change_exit)
    backtester.run_simulation()

    metrics = np.array(qs.reports.metrics(backtester.pf.returns(), index_returns, mode='full', rf=0.0437, display=False))
    strat_sortino = metrics[10][1]
    strat_alpha = metrics[67][1]

    return {"ticker": ticker,"strat_sortino": strat_sortino,"strat_alpha": strat_alpha,}

# ─── 3. Define the Tune objective ────────────────────────────────────────────

def threshold_tuner(config):
    with ThreadPoolExecutor(max_workers=len(tickers) if len(tickers) <= 10 else 10) as ex:
        metrics = ex.map(lambda t: predict_backtest(t, out_of_sample_data, raw_stock_data, index_returns, config["buy_threshold"], config["sell_threshold"], config["window_size"]), tickers)
    metrics = [metric for metric in metrics if metric is not None]
    
    mean_sortino = statistics.mean([d['strat_sortino'] for d in metrics])
    tune.report(sortino=mean_sortino)


# ─── 4. Configure search space & run ─────────────────────────────────────────

search_space = {
    "buy_threshold":  tune.uniform(0.001, 0.10),   # 0.1% → 10%
    "sell_threshold": tune.uniform(0.001, 0.10),   # 0.1% → 10%
    "window_size": tune.choice([5, 10, 20, 40, 60, 80, 100]),
}

analysis = tune.run(
    threshold_tuner,
    config     = search_space,
    metric     = "sortino",
    mode       = "max",
    num_samples= 50,        # try 50 different (buy,sell) pairs
    resources_per_trial={"cpu": 1},  # adjust GPUs/CPUs as you like
    trial_dirname_creator = lambda trial: f"{trial.trainable_name}_{trial.trial_id[:4]}"
)

best = analysis.get_best_config(metric="sortino", mode="max")
print("Best thresholds →", best)
print("Best sortino  →", analysis.best_result["sortino"])