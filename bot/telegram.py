import logger
from telegram import Update
from telegram.ext import filters
from telegram.ext import Application, MessageHandler, CallbackContext
from brain.brain import Brain
from config import settings


log = logger.get(__name__)


class TelegramBot:
    def __init__(self, brain: Brain):
        self.brain = brain
        self.app = Application.builder().token(settings.telegram_bot_token).build()

        self._register_handlers()

    def run(self):
        self.app.run_polling()

    async def handle_text(self, update: Update, context: CallbackContext):
        text_input = update.message.text
        try:
            text_output = await self.brain.think(text_input=text_input)
        except Exception as e:
            log.exception("failed to process message")
            text_output = str(e)

        await update.message.reply_text(text=text_output)

    def _register_handlers(self):
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))