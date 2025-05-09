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
import streamlit as st

from typing import List, Optional, Type, Union, Dict, Any

AgentType = Type  # or more strictly: Type[object]

class AgentManager:
    ALL_AGENTS: List[AgentType] = [
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
    AGENT_MAP: Dict[str, AgentType] = {cls.__name__: cls for cls in ALL_AGENTS}

    def __init__(
        self,
        metrics: pd.DataFrame,
        agents: Optional[List[Union[str, AgentType]]] = None,
        period: Optional[str] = 'Annual',
        streamlit_progress: Optional[bool] = False
    ):
        self.tickers: List[str] = list(metrics.ticker.values)
        self.metrics: pd.DataFrame = metrics

        if agents is None:
            self.agent_classes = self.ALL_AGENTS.copy()
        else:
            resolved: List[AgentType] = []
            for item in agents:
                if isinstance(item, str):
                    if item in self.AGENT_MAP:
                        resolved.append(self.AGENT_MAP[item])
                    else:
                        raise ValueError(
                            f"Unknown agent name '{item}'. Valid names: {list(self.AGENT_MAP.keys())}"
                        )
                elif isinstance(item, type):
                    resolved.append(item)
                else:
                    raise TypeError(
                        f"Agent entries must be class or string, got {type(item)}"
                    )
            self.agent_classes = resolved

        self.period = 'Q' if period == 'Quarterly' else 'FY'
        self.limit = 4 if period == 'Quarterly' else 10
        self.threshold_matrix_path = {'two_digit_sic':                ('/Users/thomasaltmann/PycharmProjects/Project DeepGreen/Agents/Matrices/Fundamentals Matrix - 2digit SIC.csv'),
                                      'business_services_sic':        ('/Users/thomasaltmann/PycharmProjects/Project DeepGreen/Agents/Matrices/Fundamentals Matrix - 4digit SIC 73 - Business Services.csv')
                                     }
        self.streamlit_progress = streamlit_progress

    def _analyze_one_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Run every agent for this one ticker,
        return a map agent_name → result (or exception).
        """
        results: Dict[str, Any] = {}
        for AgentCls in self.agent_classes:
            name = AgentCls.__name__
            try:
                results[name] = AgentCls(ticker, self.metrics, analysis_period=self.period, analysis_limit=self.limit, threshold_matrix_path=self.threshold_matrix_path).analyze()
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
                'Ticker': ticker,
                'Company Name': self.metrics[(self.metrics['ticker']==ticker)]['company_name'][0],
                'Bullish': bullish,
                'Bearish': bearish,
                'Neutral': neutral,
                'Score': total_score
            })

        df = pd.DataFrame(rows)
        df['Signal'] = df[['Bullish', 'Bearish', 'Neutral']].idxmax(axis=1)
        return df.set_index('Ticker')

    def agent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Parallelize over tickers: each thread runs _analyze_one_ticker.
        Returns:
          { ticker1: {AgentA: res, AgentB: res, …}, ticker2: {…}, … }
        """
        final_results: Dict[str, Dict[str, Any]] = {}
        max_workers = min(len(self.tickers), 50)
        total = len(self.tickers)
        processed = 0
        # set up Streamlit progress bar if requested
        if self.streamlit_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()

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

                if self.streamlit_progress:
                    processed += 1
                    # update progress bar as a percentage
                    progress_bar.progress(int(processed / total * 100))
                    status_text.text(f"{round(processed / total * 100,2)}% completed")
                    
        if self.streamlit_progress:
            progress_bar.empty()
            status_text.empty()
        summary_df = self._summarize(final_results)
        return final_results, summary_df