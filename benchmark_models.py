#!/usr/bin/env python
"""
Modular ML Model Benchmark Runner with W&B Integration
=====================================================

A rapid, one-shot ML development script for benchmarking multiple models
with Weights & Biases integration and LLM-generated reports.

Example
-------
$ python benchmark_models.py --experiment-run "model_comparison_v1" --models Tempus_v2 Tempus_v3 --years 1
"""
import argparse
import logging
import os
import sys
import time
import json
import importlib.util
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import quantstats_lumi as qs
from tqdm import tqdm

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from Components.TickerData import TickerData
from Components.BackTesting import BackTesting

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


class ModelBenchmarkRunner:
    """Modular benchmark runner for ML models with W&B integration"""

    def __init__(self, experiment_run: str, models: List[str], days: int = 252,
                 out_dir: str = "benchmark_results", sample_size: int = 100,
                 prediction_window: int = 3):
        self.experiment_run = experiment_run
        self.models = models
        self.days = days
        self.sample_size = sample_size
        self.prediction_window = prediction_window
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

        # Initialize W&B
        self.wandb_run = None
        #self.initialize_wandb()

        # Initialize Gemini
        self.gemini_model = "gemini-2.5-flash"
        self.initialize_gemini()

        # Model results storage
        self.model_results = {}
        self.benchmark_data = None

    def initialize_wandb(self):
        """Initialize Weights & Biases"""
        if not WANDB_AVAILABLE:
            logging.warning("⚠️  W&B not available (install with: pip install wandb)")
            return

        try:
            self.wandb_run = wandb.init(
                project="model-benchmark",
                name=self.experiment_run,
                config={
                    "models": self.models,
                    "days": self.days,
                    "experiment_run": self.experiment_run
                }
            )
            logging.info("✅ Initialized W&B run: %s", self.experiment_run)
        except Exception as e:
            logging.warning("⚠️  Could not initialize W&B: %s", e)

    def initialize_gemini(self):
        """Initialize Google Gemini API"""
        if not GEMINI_AVAILABLE:
            logging.warning("⚠️  Google Generative AI not available (install with: pip install google-generativeai)")
            return

        try:
            if not os.getenv("GEMINI_API_KEY"):
                logging.warning("⚠️  GEMINI_API_KEY not set, LLM reports disabled")
                return

            self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            logging.info("✅ Initialized Gemini API")
        except Exception as e:
            logging.warning("⚠️  Could not initialize Gemini: %s", e)

    def load_model_inference(self, model_name: str):
        """Dynamically load model inference class"""
        model_dir = Path("Models") / model_name
        inference_path = model_dir / "inference.py"

        if not inference_path.exists():
            raise FileNotFoundError(f"Inference script not found: {inference_path}")

        # Load the inference module
        spec = importlib.util.spec_from_file_location(f"{model_name}_inference", inference_path)
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)

        # Get the inference class (assumes naming convention)
        class_name = f"{model_name.replace('_', '').replace('.', '')}Inference"
        if hasattr(inference_module, class_name):
            return getattr(inference_module, class_name)()
        else:
            # Try common naming patterns
            for attr_name in dir(inference_module):
                attr = getattr(inference_module, attr_name)
                if (hasattr(attr, '__class__') and 
                    hasattr(attr, 'predict') and 
                    'Inference' in attr.__class__.__name__):
                    return attr

        raise AttributeError(f"Could not find inference class in {inference_path}")

    def load_model_datamodule(self, model_name: str, config: dict):
        """Load model-specific datamodule"""
        model_dir = Path("Models") / model_name
        datamodule_path = model_dir / "datamodule.py"

        if not datamodule_path.exists():
            logging.warning("⚠️  No datamodule found for %s, using default", model_name)
            return None

        # Load the datamodule
        spec = importlib.util.spec_from_file_location(f"{model_name}_datamodule", datamodule_path)
        datamodule_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datamodule_module)

        # Get the datamodule class
        for attr_name in dir(datamodule_module):
            attr = getattr(datamodule_module, attr_name)
            if (hasattr(attr, '__class__') and 
                hasattr(attr, 'prepare_data') and 
                'DataModule' in attr.__class__.__name__):
                return attr(config=config, days=self.days, use_cache=True)

        return None

    def prepare_benchmark_data(self):
        """Prepare benchmark data (NDX index)"""
        try:
            # Use TickerData to get benchmark data
            data_retriever = TickerData(
                ticker='I:NDX',
                indicator_list=[],
                days=self.days
            )
            
            benchmark_prices = data_retriever.preprocess_data()
            if benchmark_prices is not None and 'Close' in benchmark_prices.columns:
                benchmark_returns = benchmark_prices['Close'].pct_change().dropna()
                self.benchmark_data = benchmark_returns.cumsum().reset_index()
                self.benchmark_data.columns = ['Date', 'bench_cumulative_return']
                logging.info("✅ Loaded benchmark data (NDX)")
            else:
                logging.warning("⚠️  Could not load benchmark data")
                self.benchmark_data = None
        except Exception as e:
            logging.error("❌ Error loading benchmark data: %s", e)
            self.benchmark_data = None

    def run_model_backtest(self, model_name: str, inference_class, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for a single model"""
        logging.info("🔮 Running backtest for %s", model_name)

        # Run inference
        predictions = inference_class.predict(data)

        # Merge with price data for backtesting
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_price_cols = [col for col in price_columns if col in self.stock_data.columns]

        if not available_price_cols:
            raise ValueError(f"No OHLCV data available for backtesting in {model_name}")

        backtest_data = pd.merge(
            predictions,
            self.stock_data[available_price_cols + ['Ticker']],
            left_index=True, right_index=True, how='inner'
        )

        # Run backtests per ticker
        results = []
        returns_list = []

        tickers = backtest_data['Ticker'].unique()
        for ticker in tqdm(tickers, desc=f"Backtesting {model_name}", leave=False):
            ticker_data = backtest_data[backtest_data['Ticker'] == ticker].copy()

            if len(ticker_data) < 50:  # Skip if insufficient data
                continue

            try:
                bt = BackTesting(
                    ticker_data, ticker, initial_capital=10000,
                    pct_change_entry=0.02, pct_change_exit=0.07708,
                    use_sizing=False
                )
                bt.run_simulation()

                # Calculate metrics
                portfolio_returns = bt.pf.returns()
                if len(portfolio_returns) > 0:
                    returns_list.append(pd.DataFrame({
                        "Returns": portfolio_returns,
                        "Ticker": ticker
                    }))

                    # Calculate performance metrics
                    metrics = self.calculate_metrics(portfolio_returns)
                    metrics.update({
                        'model': model_name,
                        'ticker': ticker,
                        'backtesting_date': date.today()
                    })
                    results.append(metrics)

            except Exception as e:
                logging.warning("⚠️  Backtest failed for %s-%s: %s", model_name, ticker, e)
                continue

        # Aggregate results
        if results:
            results_df = pd.DataFrame(results)
            returns_df = pd.concat(returns_list, axis=0) if returns_list else pd.DataFrame()

            # Calculate strategy-level returns
            strategy_returns = self.aggregate_returns(returns_df) if not returns_df.empty else pd.DataFrame()

            return {
                'results_df': results_df,
                'returns_df': returns_df,
                'strategy_returns': strategy_returns
            }
        else:
            logging.warning("⚠️  No successful backtests for %s", model_name)
            return None

    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for returns series"""
        try:
            if len(returns) == 0:
                return {}

            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(returns)
            }
        except Exception as e:
            logging.warning("⚠️  Error calculating metrics: %s", e)
            return {}

    def aggregate_returns(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ticker-level returns to strategy level"""
        if returns_df.empty:
            return pd.DataFrame()

        returns_df = returns_df.reset_index().rename(columns={"index": "Date"})
        returns_df = returns_df.sort_values(['Ticker', 'Date'])
        returns_df['cum_return'] = returns_df.groupby('Ticker')['Returns'].cumsum()

        strategy_returns = (returns_df.groupby('Date')['cum_return']
                          .mean()
                          .reset_index(name='strategy_cumulative_return'))

        if 'Date' in strategy_returns.columns:
            strategy_returns['Date'] = pd.to_datetime(strategy_returns['Date'])

        return strategy_returns

    def create_comparison_plots(self) -> Dict[str, go.Figure]:
        """Create comparison plots for all models"""
        plots = {}

        # Performance comparison plot
        fig = go.Figure()

        for model_name, results in self.model_results.items():
            if results and 'strategy_returns' in results and not results['strategy_returns'].empty:
                strategy_data = results['strategy_returns']
                fig.add_trace(go.Scatter(
                    x=strategy_data['Date'],
                    y=strategy_data['strategy_cumulative_return'],
                    name=model_name,
                    mode='lines'
                ))

        # Add benchmark if available
        if self.benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=self.benchmark_data['Date'],
                y=self.benchmark_data['bench_cumulative_return'],
                name='NDX Benchmark',
                line=dict(color='grey', dash='dash')
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            yaxis_tickformat='.1%',
            height=600,
            template='plotly_white'
        )

        plots['performance_comparison'] = fig

        # Metrics comparison
        metrics_data = []
        for model_name, results in self.model_results.items():
            if results and 'results_df' in results and not results['results_df'].empty:
                model_metrics = results['results_df'].groupby('model').agg({
                    'sharpe_ratio': 'mean',
                    'total_return': 'mean',
                    'max_drawdown': 'mean',
                    'volatility': 'mean'
                }).reset_index()
                metrics_data.append(model_metrics)

        if metrics_data:
            all_metrics = pd.concat(metrics_data, ignore_index=True)

            # Create metrics comparison chart
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Volatility']
            )

            fig_metrics.add_trace(
                go.Bar(x=all_metrics['model'], y=all_metrics['sharpe_ratio'], name='Sharpe'),
                row=1, col=1
            )
            fig_metrics.add_trace(
                go.Bar(x=all_metrics['model'], y=all_metrics['total_return'], name='Return'),
                row=1, col=2
            )
            fig_metrics.add_trace(
                go.Bar(x=all_metrics['model'], y=all_metrics['max_drawdown'], name='Drawdown'),
                row=2, col=1
            )
            fig_metrics.add_trace(
                go.Bar(x=all_metrics['model'], y=all_metrics['volatility'], name='Volatility'),
                row=2, col=2
            )

            fig_metrics.update_layout(
                title='Model Metrics Comparison',
                height=600,
                showlegend=False
            )

            plots['metrics_comparison'] = fig_metrics

        return plots

    def generate_llm_report(self, summary_data: Dict[str, Any]) -> str:
        """Generate LLM narrative report using Gemini"""
        if not self.gemini_model:
            return "LLM report generation not available (Gemini API not configured)"

        try:
            # Prepare data summary for LLM
            prompt = f"""
            As a quantitative finance expert, analyze the following model benchmark results and provide a comprehensive report with recommendations.

            Experiment: {self.experiment_run}
            Models Tested: {', '.join(self.models)}
            Testing Period: {self.days} day(s)

            Performance Summary:
            {json.dumps(summary_data, indent=2, default=str)}

            Please provide:
            1. Executive Summary of findings
            2. Detailed analysis of each model's performance
            3. Comparative analysis highlighting strengths and weaknesses
            4. Risk assessment and drawdown analysis
            5. Clear recommendation on which model(s) to adopt and why
            6. Suggestions for further testing or improvements

            Format the response in clear sections with actionable insights.
            """

            response = self.genai_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt)
            return response.text

        except Exception as e:
            logging.error("❌ Error generating LLM report: %s", e)
            return f"Error generating LLM report: {e}"

    def publish_wandb_report(self, plots: Dict[str, go.Figure], llm_report: str):
        """Publish comprehensive report to W&B"""
        if not WANDB_AVAILABLE or not self.wandb_run:
            logging.info("⚠️  W&B not available, skipping report publishing")
            return

        try:
            # Log plots
            for plot_name, fig in plots.items():
                self.wandb_run.log({plot_name: wandb.Html(fig.to_html())})

            # Log summary metrics
            summary_metrics = {}
            for model_name, results in self.model_results.items():
                if results and 'results_df' in results and not results['results_df'].empty:
                    model_summary = results['results_df'].groupby('model').agg({
                        'sharpe_ratio': 'mean',
                        'total_return': 'mean',
                        'max_drawdown': 'mean'
                    }).to_dict('records')[0]

                    for metric, value in model_summary.items():
                        summary_metrics[f"{model_name}_{metric}"] = value

            self.wandb_run.log(summary_metrics)

            # Log LLM report
            self.wandb_run.log({"llm_analysis": wandb.Html(f"<pre>{llm_report}</pre>")})

            # Save artifacts
            for model_name, results in self.model_results.items():
                if results and 'results_df' in results:
                    results_path = self.out_dir / f"{model_name}_results.csv"
                    results['results_df'].to_csv(results_path, index=False)
                    self.wandb_run.log_artifact(str(results_path))

            logging.info("✅ Published report to W&B")

        except Exception as e:
            logging.error("❌ Error publishing to W&B: %s", e)

    def run_benchmark(self):
        """Run the complete benchmark process"""
        logging.info("🚀 Starting benchmark run: %s", self.experiment_run)

        # Prepare benchmark data
        self.prepare_benchmark_data()

        # Prepare the initial sample set
        data_retriever = TickerData(
            indicator_list=None,
            days=self.days,
            prediction_window=self.prediction_window,
            prediction_mode=True,
            sample_size=self.sample_size
        )
        self.stock_data = data_retriever.preprocess_data()
        logging.info("✅ Finished pulling initial OHLCV data shared among models")

        print(self.stock_data)

        # Run each model
        for model_name in self.models:
            try:
                logging.info("📊 Processing model: %s", model_name)

                # Load model components
                inference_class = self.load_model_inference(model_name)
                datamodule = self.load_model_datamodule(model_name, inference_class.constants)

                # Prepare data
                if datamodule:
                    data = datamodule.prepare_data(self.stock_data)
                    
                    if not data:
                        logging.warning("⚠️  No data available for %s", model_name)
                        continue
                else:
                    logging.warning("⚠️  No datamodule available for %s", model_name)
                    continue

                # Run backtest
                results = self.run_model_backtest(model_name, inference_class, data)
                self.model_results[model_name] = results

                logging.info("✅ Completed %s", model_name)

            except Exception as e:
                logging.error("❌ Error processing %s: %s", model_name, e)
                self.model_results[model_name] = None

        # Generate comparison plots
        plots = self.create_comparison_plots()

        # Save plots locally
        for plot_name, fig in plots.items():
            plot_path = self.out_dir / f"{plot_name}.html"
            fig.write_html(plot_path)
            logging.info("💾 Saved plot: %s", plot_path)

        # Generate LLM report
        summary_data = {
            model_name: {
                'avg_sharpe': results['results_df']['sharpe_ratio'].mean() if results and 'results_df' in results and not results['results_df'].empty else 0,
                'avg_return': results['results_df']['total_return'].mean() if results and 'results_df' in results and not results['results_df'].empty else 0,
                'max_drawdown': results['results_df']['max_drawdown'].mean() if results and 'results_df' in results and not results['results_df'].empty else 0
            }
            for model_name, results in self.model_results.items()
        }

        llm_report = self.generate_llm_report(summary_data)

        # Save LLM report
        report_path = self.out_dir / "llm_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(llm_report)
        logging.info("💾 Saved LLM report: %s", report_path)

        # Publish to W&B
        #self.publish_wandb_report(plots, llm_report)

        # Save summary
        summary_path = self.out_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'experiment_run': self.experiment_run,
                'models': self.models,
                'summary_data': summary_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)

        logging.info("✅ Benchmark complete! Results saved to %s", self.out_dir)

        if self.wandb_run:
            self.wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Modular ML Model Benchmark Runner")
    #parser.add_argument("--experiment-run", required=True,help="W&B experiment run name")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model names to benchmark (e.g., Tempus_v2 Tempus_v3)")
    parser.add_argument("--days", type=int, default=252,
                       help="Days of data to use for backtesting")
    parser.add_argument("--horizon", default=3,
                   help="Forecast horizon for models")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--out-dir", default="benchmark_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Run benchmark
    runner = ModelBenchmarkRunner(
        experiment_run=None,
        models=args.models,
        days=args.days,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        prediction_window=args.horizon
    )

    runner.run_benchmark()


if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info("Total runtime %.1f s", time.time() - t0)