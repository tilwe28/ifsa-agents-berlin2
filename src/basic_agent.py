import random

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType


class BasicAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        # Override verify_market method from DeployableTraderAgent, otherwise it would do LLM calls to OpenAI to verify the market's question predictability.
        # Overrided so that we can run this super simple agent without OpenAI API key.
        return True

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        decision = random.choice([True, False])
        return ProbabilisticAnswer(
            confidence=0.5,
            p_yes=Probability(float(decision)),
            reasoning="I flipped a coin to decide.",
        )


if __name__ == "__main__":
    agent = BasicAgent()
    agent.run(market_type=MarketType.OMEN)
