import os
import logger
import logging
from config import settings
from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import set_tracing_disabled
from bot.telegram import TelegramBot
from brain.brain import Brain


os.environ["OPENVIKING_CONFIG_FILE"] = settings.openviking_config_file

logger.init(level=logging.DEBUG)
log = logger.get(__name__)

set_default_openai_client(AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key))
set_tracing_disabled(True)


if __name__ == "__main__":
    brain = Brain()
    brain.introspect()

    bot = TelegramBot(brain=brain)
    bot.run()
