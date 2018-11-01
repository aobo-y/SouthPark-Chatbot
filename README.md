# SouthPark Chatbot

## Setup

The code requires `python3`. Install dependencies by

```
pip install -r requirements.txt
```

## Usage

*If you want to run it on the school server, run the following command to load the some modules first*

```bash
source server_env
```

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

### Tune parameters

The following model parameters in the `config.py` file can be tuned:<br>
* NUM_LAYERS: the number of GRU/LSTM layers
* USE_LSTM: use GRU or LSTM cell
* EMBEDDING_SIZE: the length of the embedding vector for each symbol
* ATTENTION_HEADS: the number of hidden states that read from the attention state
* BATCH_SIZE: the size of training data per iteration
* LR: learning rate of the model

