import os
import threading
from openai import AsyncOpenAI
from agents import set_default_openai_client
from agents import set_tracing_disabled
from brain.brain import Brain
from config import settings, clear_openviking_logger


os.environ["OPENVIKING_CONFIG_FILE"] = settings.openviking_config_file
clear_openviking_logger()

set_default_openai_client(AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key))
set_tracing_disabled(True)


def _introspect_loop(brain: Brain, interval: float, stop: threading.Event):
    while not stop.wait(interval):
        try:
            brain.introspect()
        except Exception as e:
            print(f"  introspect | error: {e}")


if __name__ == "__main__":
    brain = Brain()
    stop_event = threading.Event()

    t = threading.Thread(target=_introspect_loop, args=(brain, settings.reflection_interval, stop_event), daemon=True)
    t.start()

    try:
        while True:
            text_input = input("> ")
            text_output = brain.think(text_input=text_input)
            print(f"\n[soul]\n{text_output}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        brain.close()
