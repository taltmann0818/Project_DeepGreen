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
        tickers: List[str],
        metrics: pd.DataFrame,
        agents: Optional[List[AgentType]] = None
    ):
        self.tickers: List[str] = tickers
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

    def _analyze_one_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Run every agent for this one ticker,
        return a map agent_name → result (or exception).
        """
        results: Dict[str, Any] = {}
        for AgentCls in self.agent_classes:
            name = AgentCls.__name__
            try:
                results[name] = AgentCls(ticker, self.metrics).analyze()
            except Exception as e:
                results[name] = e
        return results

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

        return final_results