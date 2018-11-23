# -*- coding: utf-8 -*-

# parameters for processing the dataset
MAX_LENGTH = 20  # Maximum sentence length to consider
MIN_COUNT = 1    # Minimum word count threshold for trimming

# Default word tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

# corpus information
CORPUS_NAME = "south_park"
CORPUS_FILE = "fine_tune.txt"

# Configure models
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'dwy_persona_based' 
LOAD_CHECKPOINT = False   
CHECKPOINT_ITER = 100      # where to continue training
ATTN_MODEL = 'dot'         # type of the attention model: dot/general/concat
TRAIN_EMBEDDING = True     # whether to update the word embeddding during training
USE_PERSONA = True         # whether to update the persona embedding during training
HIDDEN_SIZE = 500          # number of hidden units in bi-GRU encoder
PERSONA_SIZE = 100         # size of the persona embedding
ENCODER_N_LAYERS = 2       # number of layers in bi-GRU encoder
DECODER_N_LAYERS = 2       # number of layers in GRU decoder
ENCODER_DROPOUT_RATE = 0.1 # dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE = 0.1 # dropout rate in GRU decoder
BATCH_SIZE = 64            # size of the mini batch in training state

# Configure training/optimization
N_ITER = 100               # training iterations
CLIP = 50.0                # gradient norm clip
TEACHER_FORCING_RATIO = 1.0
LR = 0.0001                # encoder learning ratio
DECODER_LR = 5.0           # decoder learning ratio: LR*DECODER_LR
PRINT_EVERY = 10           # print the loss every x iterations
SAVE_EVERY = 100           # save the checkpoint every x iterations

