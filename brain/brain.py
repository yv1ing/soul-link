from agents import Agent, Runner
from config import settings
from memory import HybridMemory


class Brain:
    def __init__(self, session_id: str = "default"):
        self.agent = Agent(
            name="Soul brain agent",
            model=settings.brain_model,
            instructions=settings.brain_prompt,
        )
        self.memory = HybridMemory(session_id=session_id)

    def think(self, text_input: str) -> str:
        result = Runner.run_sync(
            starting_agent=self.agent,
            input=text_input,
            session=self.memory,
        )
        return result.final_output

    def close(self):
        self.memory.close()