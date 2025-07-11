# ~/ai_agent_system/telegram_bot.py
import os
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

import database
import agent_definitions
import orchestrator

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я агентская система для поиска и анализа информации. "
        "Отправь мне свой запрос, чтобы начать.\n\n"
        "Я буду кратко транслировать весь путь выполнения задачи."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text
    chat_id = update.message.chat_id

    logger.info(f"Received query from chat_id {chat_id}: {user_query}")

    asyncio.create_task(
        orchestrator.run_full_agent_process(user_query, chat_id, context.bot.send_message)
    )
    await update.message.reply_text("Ваш запрос принят в работу. Пожалуйста, ожидайте, это может занять некоторое время...")

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в .env")
        print("Ошибка: TELEGRAM_BOT_TOKEN не установлен. Пожалуйста, проверьте файл .env")
        return

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен. Ожидаю сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    database.init_db()
    agent_definitions.define_agents()
    main()
