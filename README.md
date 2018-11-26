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

## Train

The model can restore the previously trained weights and continue training up on that. 

```bash
cd src
python3 chatbot.py --mode train
```

## Chat

Start the command line interaction with the chat bot.

```bash
cd src
python3 chatbot.py --mode chat --speaker cartman
```

## Config

The model settings can be modified in `src/config.py`

Parameters | Description
-----|------
DATA_MODE | choose which data set to train on: general_data/persona_data
MAX_LENGTH | maximum sentence length to consider
MIN_COUNT | minimum word count threshold for trimming
N_ITER | training iterations
LOAD_CHECKPOINT | whether to load the checkpoints, if true, need to set CHECKPOINT_ITER
ATTN_MODEL | type of the attention model: dot/general/concat
TRAIN_EMBEDDING | whether to update the word embeddding during training
USE_PERSONA | whether to update the persona embedding during training
HIDDEN_SIZE | size of the word embedding & number of hidden units in GRU
PERSONA_SIZE | size of the persona embedding
ENCODER_N_LAYERS | number of layers in bi-GRU encoder
DECODER_N_LAYERS | number of layers in GRU decoder
ENCODER_DROPOUT_RATE | dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE | dropout rate in GRU decoder
TEACHER_FORCING_RATIO | the ratio that decoder learns from ground truth instead of last output
LR | encoder learning rate
DECODER_LR | decoder learning rate: LR*DECODER_LR