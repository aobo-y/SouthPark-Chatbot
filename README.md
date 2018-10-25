# SouthPark Chatbot

## Usage

### Chat CLI

Start the command line interaction with the chat bot.
```
python src/chatbot.py --mode chat
```

### Train

The model will restore the previously trained weights and continue training up on that. To start training from scratch, please delete all the checkpoints in the checkpoints folder.

```
python src/chatbot.py --mode train
```

