from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import chatbot
from evaluate import evaluateExample

speakers = {
    'cartman': '771434365:AAE8bpXZ6QMizyZxmvr6Mke6pOUFW0suN9E',
    'stan': '797666745:AAG5L9qHQgQIERXFf47mD1QdatLEfXH0p2c',
    'kyle': '798154398:AAGHAUZ8l-Zi_bMr8KrRwkZ1svju1_fW2S8',
    'randy': '764465736:AAHeOYQMHStGNPZ1gNaPl2dZEiuorvBi_fI',
    'butters': '789795577:AAHOF3l49Ij-pqsAkEp0Xw84bMTn1ui4lxo',
    '<none>': '773092951:AAHMltKlernAmXHvO_TQ3B6mzkY1mv61rQc'
}


searcher, word_map, person_map = chatbot.init()

class TeleBot:
    def __init__(self, speaker, token):
        self.speaker = speaker
        self.token = token
    def run(self):
        updater = Updater(token=self.token)
        dispatcher = updater.dispatcher
        speaker_id = person_map.get_index(self.speaker)
        def start(bot, update):
            bot.send_message(chat_id=update.message.chat_id, text=evaluateExample(
                    "",
                    searcher,
                    word_map,
                    speaker_id))
        start_handler = CommandHandler('start', start)
        dispatcher.add_handler(start_handler)
        def response(bot, update):
            bot.send_message(
                chat_id=update.message.chat_id,
                text=evaluateExample(
                    update.message.text,
                    searcher,
                    word_map,
                    speaker_id))
        message_handler = MessageHandler(Filters.text, response)
        dispatcher.add_handler(message_handler)
        updater.start_polling()
        print(f'Telegram bot {self.speaker} has started')

if __name__ == '__main__':
    for speaker, token in speakers.items():
        bot = TeleBot(speaker, token)
        bot.run()



