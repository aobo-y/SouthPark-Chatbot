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

The model will restore the previously trained weights and continue training up on that. To start training from scratch, please delete all the checkpoints in the checkpoints folder.

```bash
cd src
python3 chatbot.py --mode train
```

## Chat

Start the command line interaction with the chat bot.

```bash
cd src
python3 chatbot.py --mode chat
```

## Config

The model parameters can be tuned in 'src/config.py'
