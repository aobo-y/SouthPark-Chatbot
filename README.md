# SouthPark Chatbot

## Setup

The code requires `python3`. Install dependencies by

```
pip install -r requirements.txt
```

## Usage

*If you want to run it on the school server, run the following command to load the some modules first*

```bash
source scripts/server_env
python3 src/chatbot.py --mode train # train on school server
python3 src/chatbot.py --mode chat # chat on school server
```

### Chat CLI

Start the command line interaction with the chat bot.
```bash
python src/chatbot.py --mode chat
```

### Train

The model will restore the previously trained weights and continue training up on that. To start training from scratch, please delete all the checkpoints in the checkpoints folder.

```bash
python src/chatbot.py --mode train
```

### Config

The following model parameters can be tuned in `config.py`

Parameters | Description
-----|------
NUM_LAYERS | the number of GRU/LSTM layers
USE_LSTM | use GRU or LSTM cell
USE_DROPOUT | use dropout layer or not
DROPOUT_INPUT_KEPP_PROB | [0, 1], input keep probability
DROPOUT_OUTPUT_KEEP_PROB | [0, 1], output keep probability
DROPOUT_STATE_KEPP_PROB | [0, 1], state dropout is performed on the outgoing states of the cell
EMBEDDING_SIZE | the length of the embedding vector for each symbol
ATTENTION_HEADS | the number of hidden states that read from the attention state
BATCH_SIZE | the size of training data per iteration
LR | learning rate of the model

