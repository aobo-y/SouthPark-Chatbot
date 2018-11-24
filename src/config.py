# -*- coding: utf-8 -*-

# Corpus information
# different mode: cornell/pretrain/finetune
DATA_MODE = 'cornell'  
# pretrain on cornell movie
if DATA_MODE == 'cornell':
    CORPUS_NAME = "cornell movie-dialogs corpus"
    CORPUS_NAME_PRETRAIN = "cornell movie-dialogs corpus"
    CORPUS_FILE = "formatted_movie_lines.txt"
# pretrain on south park general
if DATA_MODE == 'pretrain':
    CORPUS_NAME = "south_park"
    CORPUS_NAME_PRETRAIN = "cornell movie-dialogs corpus"
    CORPUS_FILE = "general_train.txt"
# fine tune on south park persona
if DATA_MODE == 'finetune':
    CORPUS_NAME = "south_park"
    CORPUS_NAME_PRETRAIN = "cornell movie-dialogs corpus"
    CORPUS_FILE = "fine_tune.txt"


# Parameters for processing the dataset
MAX_LENGTH = 20  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming
PAD_TOKEN = 0    # Padding token id
SOS_TOKEN = 1    # Start of Sentence token id
EOS_TOKEN = 2    # End of Sentence token id
UNK_TOKEN = 3    # Unknown token id


# Configure models
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'dwy_persona_based'
N_ITER = 4000                # training iterations 
LOAD_CHECKPOINT = False      # whether to load checkpoint, if true, need to set CHECKPOINT_ITER
CHECKPOINT_ITER = 4000       # where to continue training
ATTN_MODEL = 'dot'           # type of the attention model: dot/general/concat
TRAIN_EMBEDDING = True       # whether to update the word embeddding during training
USE_PERSONA = False          # whether to update the persona embedding during training
HIDDEN_SIZE = 500            # size of the word embedding & number of hidden units in GRU
PERSONA_SIZE = 100           # size of the persona embedding
ENCODER_N_LAYERS = 2         # number of layers in bi-GRU encoder
DECODER_N_LAYERS = 2         # number of layers in GRU decoder
ENCODER_DROPOUT_RATE = 0.1   # dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE = 0.1   # dropout rate in GRU decoder
TEACHER_FORCING_RATIO = 1.0  # ratio for training decoder on ground truth or last output of decoder
BEAM_SEARCH_ON = False     # use Beam Search or Greedy Search

# Configure training/optimization
BATCH_SIZE = 64            # size of the mini batch in training state
CLIP = 50.0                # gradient norm clip
LR = 0.0001                # encoder learning ratio
DECODER_LR = 5.0           # decoder learning ratio: LR*DECODER_LR
PRINT_EVERY = 100          # print the loss every x iterations
SAVE_EVERY = 1000          # save the checkpoint every x iterations
