from Agents.ben_graham import BenGrahamAgent
from Agents.bill_ackman import BillAckmanAgent
from Agents.cathie_wood import CathieWoodAgent
from Agents.charlie_munger import CharlieMungerAgent
from Agents.peter_lynch import PeterLynchAgent
from Agents.phil_fisher import PhilFisherAgent
from Agents.stanley_druckenmiller import StanleyDruckenmillerAgent
from Agents.warren_buffett import WarrenBuffettAgent
from Agents.valuation import ValuationAgent
from Agents.fundamentals import FundamentalsAgent

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from typing import List, Optional, Type, Dict, Any

AgentType = Type  # or more strictly: Type[object] if you want

class AgentManager:
    def __init__(
        self,
        metrics: pd.DataFrame,
        agents: Optional[List[AgentType]] = None,
        period: Optional[str] = 'Annual',
    ):
        self.tickers: List[str] = list(metrics.ticker.values)
        self.metrics: pd.DataFrame = metrics
        # allow injection or default to all agent classes
        self.agent_classes: List[AgentType] = agents or [
            BenGrahamAgent,
            BillAckmanAgent,
            CathieWoodAgent,
            CharlieMungerAgent,
            PeterLynchAgent,
            PhilFisherAgent,
            StanleyDruckenmillerAgent,
            WarrenBuffettAgent,
            ValuationAgent,
            FundamentalsAgent,
        ]
        self.period = 'Q' if period == 'Quarterly' else 'FY'
        self.limit = 4 if period == 'Quarterly' else 10

    def _analyze_one_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Run every agent for this one ticker,
        return a map agent_name → result (or exception).
        """
        results: Dict[str, Any] = {}
        for AgentCls in self.agent_classes:
            name = AgentCls.__name__
            try:
                results[name] = AgentCls(ticker, self.metrics, analysis_period=self.period, analysis_limit=self.limit).analyze()
            except Exception as e:
                results[name] = e
        return results

    def _summarize(self, raw: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Given the nested dict { ticker: {agent_name: result, …}, … },
        build a DataFrame with counts of each signal and total score.
        """
        rows = []
        for ticker, agents_out in raw.items():
            bullish = bearish = neutral = 0
            total_score = 0.0
            for res in agents_out.values():
                # skip agents that raised exceptions
                if isinstance(res, Exception):
                    continue
                sig = res.get('signal', None)
                if sig == 'bullish':
                    bullish += 1
                elif sig == 'bearish':
                    bearish += 1
                else:
                    neutral += 1
                # assume score is numeric, defaulting to 0
                total_score += float(res.get('score', 0))
            rows.append({
                'ticker': ticker,
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'score': total_score
            })

        df = pd.DataFrame(rows)
        df['signal'] = df[['bullish', 'bearish', 'neutral']].idxmax(axis=1)
        return df.set_index('ticker')

    def agent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Parallelize over tickers: each thread runs _analyze_one_ticker.
        Returns:
          { ticker1: {AgentA: res, AgentB: res, …}, ticker2: {…}, … }
        """
        final_results: Dict[str, Dict[str, Any]] = {}
        max_workers = min(len(self.tickers), 50)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._analyze_one_ticker, tk): tk
                for tk in self.tickers
            }

            for future in as_completed(future_to_ticker):
                tk = future_to_ticker[future]
                try:
                    final_results[tk] = future.result()
                except Exception as e:
                    final_results[tk] = {"_fatal_error": e}

        summary_df = self._summarize(final_results)
        return final_results, summary_df