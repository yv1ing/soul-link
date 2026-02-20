import os
from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import set_tracing_disabled
from brain.brain import Brain
from config import settings, clear_openviking_logger


os.environ["OPENVIKING_CONFIG_FILE"] = settings.openviking_config_file
clear_openviking_logger()

set_default_openai_client(AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key))
set_tracing_disabled(True)

if __name__ == "__main__":
    brain = Brain()

    try:
        while True:
            text_input = input("> ")
            text_output = brain.think(text_input=text_input)
            print(text_output)

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        brain.close()
