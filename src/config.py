# -*- coding: utf-8 -*-

# parameters for processing the dataset
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming

# Default word tokens
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

# corpus information
CORPUS_NAME = "cornell movie-dialogs corpus"
CORPUS_FILE = "formatted_movie_lines.txt"

# Configure models
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'dwy' 
LOAD_CHECKPOINT = False   
CHECKPOINT_ITER = 4000     # where to continue training
ATTN_MODEL = 'dot'         # type of the attention model: dot/general/concat
HIDDEN_SIZE = 500          # number of hidden units in bi-GRU encoder
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
PRINT_EVERY = 1            # print the loss every x iterations
SAVE_EVERY = 10            # save the checkpoint every x iterations

