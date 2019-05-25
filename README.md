# SouthPark Chatbot

## Try it on Telegram

Our bots are running on telegram

 Profile | Name | id
 -|--|--
 <img width="50" alt="cartman" src="doc/cartman.png"> | Eric Cartman | sp_cartman_bot
 <img width="50" alt="stan" src="doc/stan.png"> | Stan Marsh | sp_stan_bot
 <img width="50" alt="kyle" src="doc/kyle.png"> | Kyle Broflovski | sp_kyle_bot
 <img width="50" alt="butters" src="doc/butters.png"> | Butters Stotch | sp_butters_bot
 <img width="50" alt="randy" src="doc/randy.png"> | Randy Marsh | sp_randy_bot
 <img width="50" alt="mysterion" src="doc/mysterion.png"> | Mysterion | sp_mysterion_bot

## Setup

The code requires `python3`. Install dependencies by

```
pip install -r requirements.txt
```

## Usage


### Pretrain

Pretrain the model with anonymous dialogues, which keeps the initial personas stable.

The model can restore the previously trained weights and continue training up on that.

```bash
python3 src/chatbot.py --mode=pretrain
```

### Finetune

Finetune the model with personalized dialogues, which aims to train the personas.

```bash
python3 src/chatbot.py --mode=finetune --checkpoint=pretrain_1000
```

### Chat

Start the command line interaction with the chat bot.

```bash
python3 src/chatbot.py --mode=chat --checkpoint=finetune_1000 --speaker=cartman
```

## Config

The model settings can be modified in `src/config.py`

Parameters | Description
-----|------
DATA_MODE | choose which data set to train on: general_data/persona_data
MAX_LENGTH | maximum sentence length to consider
N_ITER | training iterations
RNN_TYPE | model of encoder & decode, support `LSTM` & `GRU`
ATTN_TYPE | type of the attention model: dot/general/concat
HIDDEN_SIZE | size of the word embedding & number of hidden units in GRU
PERSONA_EMBEDDING_SIZE | size of the persona embedding
ENCODER_N_LAYERS | number of layers in bi-GRU encoder
DECODER_N_LAYERS | number of layers in GRU decoder
ENCODER_DROPOUT_RATE | dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE | dropout rate in GRU decoder
TEACHER_FORCING_RATIO | the ratio that decoder learns from ground truth instead of last output
LR | encoder learning rate
DECODER_LR | decoder learning rate: LR*DECODER_LR
TF_RATE_DECAY_FACTOR | k in the inverse sigmoid decay func of the teacher force rate



## Miscellaneous

*If you want to run it on the school server, run the following command to load the some modules first*

```bash
source server_env
```
