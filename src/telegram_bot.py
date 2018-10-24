from chatbot import Chatbot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

spbot = Chatbot()

updater = Updater(token='771434365:AAE8bpXZ6QMizyZxmvr6Mke6pOUFW0suN9E')

dispatcher = updater.dispatcher

def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="I'm Cartman!")

def response(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text=spbot.response(update.message.text))

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

message_handler = MessageHandler(Filters.text, response)
dispatcher.add_handler(message_handler)

updater.start_polling()

print('Telegram bot has started')

