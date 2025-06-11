import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from inference_pipeline import get_answer

TOKEN = "<token>"

# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Я чат-бот техподдержки. Задайте мне вопрос."
    )

async def request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    answer = get_answer(question)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=answer
    )

    
if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    
    request_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), request)
    start_handler = CommandHandler('start', start)
    
    application.add_handler(start_handler)
    application.add_handler(request_handler)
    application.run_polling()