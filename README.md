# SouthPark Chatbot

## Setup

The code requires `python3`. Install dependencies by

```
pip install -r requirements.txt
```

## Usage

*If you want to run it on the school server, load the following modules first*

```bash
module load python3
module load anaconda3
module load cuda-toolkit
module load cudnn
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

