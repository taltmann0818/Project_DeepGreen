#!/usr/bin/env python
"""
One-shot CI benchmark runner
===========================

Example
-------
$ python benchmark_models.py --models Tempus_v2 --sample-size 100 --out-dir artefacts --sharpe-min 1.0
"""
import argparse, logging, os, random, sys, time
from tqdm import tqdm
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from polygon import RESTClient
import plotly.graph_objects as go
import quantstats_lumi as qs

from Components.TickerData import TickerData
from Components.ModelInference import onnx_predict
from Components.BackTesting import BackTesting

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
_INDEX_CONFIG = {  # table-index, column-index per Wikipedia page
    'NASDAQ':        ('https://en.wikipedia.org/wiki/Nasdaq-100',                 4, 1),
    'S&P500':        ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',0, 0),
    'RUSSELL1000':   ('https://en.wikipedia.org/wiki/Russell_1000_Index',        3, 1),
    'DOWJONES':      ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',2, 2),
}


# ──────────────────────────────────────────────────────────────────────────────
def get_index_tickers(indices, sample_size):
    """Return a de-duplicated <sample_size> random ticker list."""
    tickers = []
    for idx in indices:
        url, tbl, col = _INDEX_CONFIG[idx]
        df = pd.read_html(url)[tbl]
        tickers.extend(df.iloc[:, col].dropna().astype(str).tolist())
    tickers = list(dict.fromkeys(tickers))          # dedupe, keep order
    if sample_size > len(tickers):
        raise ValueError("sample_size larger than available tickers")
    return random.sample(tickers, sample_size)


def score_model(model_name, processed, raw, index_returns,
                entry_th=0.02, exit_th=0.07708,
                initial_capital=10_000, rf=0.04236,
               *, show_bar=True):
    """Run the full back-test for one ONNX model; return (summary_row, returns_df)."""
    rows, returns = [], []
    tickers = processed['Ticker'].unique()
    loop = tqdm(tickers,
                desc=f"🔮  {model_name}",
                leave=False, position=1) if show_bar else tickers
    
    for ticker in loop:
        preds_df = onnx_predict(f"Models/{model_name}.onnx",
                                processed[processed['Ticker'] == ticker],
                                window_size=20)
        preds_df = pd.merge(
            preds_df,
            raw[raw['Ticker'] == ticker][['open', 'high', 'low', 'volume', 'close']],
            left_index=True, right_index=True, how='left'
        )

        bt = BackTesting(
            preds_df.rename(columns={'open':'Open','high':'High',
                                     'low':'Low','close':'Close',
                                     'volume':'Volume'}),
            ticker, initial_capital, entry_th, exit_th, use_sizing=False)
        bt.run_simulation()
        returns.append(pd.DataFrame({"Returns": bt.pf.returns(),
                                     "Ticker":  ticker}))

        m = np.array(qs.reports.metrics(bt.pf.returns(),
                                        index_returns, mode='full',
                                        rf=rf, display=False))

        rows.append(dict(
            backtesting_date=date.today(), model=model_name, ticker=ticker,
            cumReturn=m[4][1], CAGR=m[5][1],
            Sharpe=m[10][1], Sortino=m[12][1],
            MaxDrawdown=m[16][1], dVaR=m[27][1],
            Alpha=m[58][1], Beta=m[57][1]
        ))
        logging.info("%s | %s done", model_name, ticker)
    return pd.DataFrame(rows), pd.concat(returns, axis=0)


def aggregate_returns(returns_all):
    """Collapse ticker-level returns to strategy-level cumulative series."""
    returns_all = returns_all.reset_index().rename(columns={"index":"Date"})
    returns_all = returns_all.sort_values(['Ticker', 'Date'])
    returns_all['cum_return'] = returns_all.groupby('Ticker')['Returns'].cumsum()
    strat = (returns_all.groupby('Date')['cum_return']
                      .mean()
                      .reset_index(name='strat_cumulative_return'))
    strat['Date'] = strat['Date'].dt.tz_localize(None)
    return strat


def plot(strat, bench, title, path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strat['Date'], y=strat['strat_cumulative_return'],
        name=title))
    fig.add_trace(go.Scatter(
        x=bench['Date'], y=bench['bench_cumulative_return'],
        name='NDX Benchmark', line=dict(color='grey')))
    fig.add_hline(y=0,line_dash="dash",line_color="black",line_width=2)
    fig.update_layout(title='Model Backesting Summary',
                      xaxis_title="Date",
                      yaxis_title="Cumulative Return %",
                      yaxis_tickformat=".1%",
                      height=600,
                      template='ggplot2',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02)
                     )
    fig.write_html(path)
    logging.info("⬆️  wrote %s", path)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="CI benchmark runner")
    p.add_argument("--models", nargs="+", required=True,
                   help="model names (without .onnx)")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--out-dir", default="artefacts")
    p.add_argument("--indices", nargs="+",
                   default=["S&P500", "RUSSELL1000", "DOWJONES"])
    p.add_argument("--sharpe-min", type=float, default=1.0,
                   help="CI fails if Sharpe < this threshold")
    cfg = p.parse_args()

    api_key = "XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2" #os.getenv("POLYGON_API_KEY")
    if not api_key:
        logging.error("Set POLYGON_API_KEY env var"); sys.exit(1)

    out = Path(cfg.out_dir); out.mkdir(exist_ok=True)

    # 1) Universe
    tickers = get_index_tickers(cfg.indices, cfg.sample_size)
    logging.info("Back-testing on %d tickers: %s …", len(tickers), ", ".join(tickers[:10]))

    # 2) Data pull (once!)
    client = RESTClient(api_key, num_pools=50)
    ind = ['ema_20','ema_50','ema_100','stoch_rsi14','macd','hmm_state','Close']
    td = TickerData(tickers, ind, client=client)
    processed, raw = td.process_all()

    # index benchmark
    index_ret = TickerData('I:NDX', [], client).get_ohlc_for_ticker('I:NDX')['close'].pct_change().tz_localize(None)
    bench_df = index_ret.cumsum().reset_index().rename(columns={"date":"Date",
                                                                "close":"bench_cumulative_return"})

    # 3) Run each model
    summaries = []
    for m in tqdm(cfg.models, desc="🏗️  Models", position=0):
        s_df, r_df = score_model(m, processed, raw, index_ret, show_bar=True)
        summaries.append(s_df)
        strat_curve = aggregate_returns(r_df)
        plot(strat_curve, bench_df, f"{m} cumulative return", out/f"{m}_plot.html")

        # CI pass/fail rule
        if s_df['Sharpe'].mean() < cfg.sharpe_min:
            logging.error("%s failed Sharpe threshold (%.2f < %.2f)",
                          m, s_df['Sharpe'].mean(), cfg.sharpe_min)
            exit_code = 1
        else:
            exit_code = 0

    summary = pd.concat(summaries).sort_values(["model", "ticker"])
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    logging.info("📄  wrote %s", out/"benchmark_summary.csv")
    sys.exit(exit_code)


if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info("Total runtime %.1f s", time.time() - t0)
