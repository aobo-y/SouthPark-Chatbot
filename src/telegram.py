from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

speakers = {
    'cartman': '771434365:AAE8bpXZ6QMizyZxmvr6Mke6pOUFW0suN9E',
    'stan': '797666745:AAG5L9qHQgQIERXFf47mD1QdatLEfXH0p2c',
    'kyle': '798154398:AAGHAUZ8l-Zi_bMr8KrRwkZ1svju1_fW2S8',
    'randy': '764465736:AAHeOYQMHStGNPZ1gNaPl2dZEiuorvBi_fI',
    'none': '773092951:AAHMltKlernAmXHvO_TQ3B6mzkY1mv61rQc'
}

updater = Updater(token='771434365:AAE8bpXZ6QMizyZxmvr6Mke6pOUFW0suN9E')

dispatcher = updater.dispatcher

def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="I'm Cartman!")

def response(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text=update.message.text)

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

message_handler = MessageHandler(Filters.text, response)
dispatcher.add_handler(message_handler)

updater.start_polling()

print('Telegram bot has started')
