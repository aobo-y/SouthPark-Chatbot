from flask import Flask, request
from chatbot import Chatbot

bot = Chatbot()

app = Flask('__name__')

@app.route("/", methods=['GET'])
def index():
    return "SouthPark Charbot Service"

@app.route("/chat", methods=['POST'])
def chat():
    body = request.get_json()
    if body is None or body['input'] is None:
        return 'empty request body', 400

    return bot.response(body['input'])

if __name__ == '__main__':
    app.run(host='0.0.0.0')
