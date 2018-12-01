# -*- coding: utf-8 -*-

# Corpus information
# different mode: general_data/persona_data
DATA_MODE = 'general_data'
# pretrain on cornell movie and south park general
if DATA_MODE == 'general_data':
    CORPUS_NAME = "general_data"
    CORPUS_NAME_PRETRAIN = "general_data"
    CORPUS_FILE = "train.txt"
# fine tune on south park persona
if DATA_MODE == 'persona_data':
    CORPUS_NAME = "persona_data"
    CORPUS_NAME_PRETRAIN = "general_data"
    CORPUS_FILE = "train.txt"

# Word Embedding
WORD_EMBEDDING_FILES = [
    'data/word_embedding/filtered.glove.42B.300d.part1.txt',
    'data/word_embedding/filtered.glove.42B.300d.part2.txt'
]


# Parameters for processing the dataset
MAX_LENGTH = 20  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming

SPECIAL_WORD_EMBEDDING_TOKENS = {
    'PAD': '<pad>', # Padding token
    'SOS': '<sos>', # Start of Sentence token
    'EOS': '<eos>', # End of Sentence token
    'UNK': '<unk>' # pretrained word embedding usually has this
}

PERSONS = ['kyle', 'cartman', 'stan', 'chef', 'kenny', 'mr. garrison',
           'randy', 'sharon', 'gerald', 'butters']

NONE_PERSONA = '<none>'

# Configure models
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'dwy_persona_based'
N_ITER = 10000               # training iterations
LOAD_CHECKPOINT = False      # whether to load checkpoint, if true, need to set CHECKPOINT_ITER
CHECKPOINT_ITER = 10000      # where to continue training
ATTN_MODEL = 'dot'           # type of the attention model: dot/general/concat
TRAIN_EMBEDDING = True       # whether to update the word embeddding during training
USE_PERSONA = False          # whether to update the persona embedding during training
HIDDEN_SIZE = 300            # size of the word embedding & number of hidden units in GRU
PERSONA_SIZE = 100           # size of the persona embedding
ENCODER_N_LAYERS = 2         # number of layers in bi-GRU encoder
DECODER_N_LAYERS = 2         # number of layers in GRU decoder
ENCODER_DROPOUT_RATE = 0.1   # dropout rate in bi-GRU encoder
DECODER_DROPOUT_RATE = 0.1   # dropout rate in GRU decoder
TEACHER_FORCING_RATIO = 1.0  # ratio for training decoder on ground truth or last output of decoder
BEAM_SEARCH_ON = False       # use Beam Search or Greedy Search
BEAM_WIDTH = 10              # size of beam
RNN_TYPE = 'LSTM'              # use LSTM or GRU as RNN

# Configure training/optimization
BATCH_SIZE = 64            # size of the mini batch in training state
CLIP = 50.0                # gradient norm clip
LR = 0.0001                # encoder learning ratio
DECODER_LR = 5.0           # decoder learning ratio: LR*DECODER_LR
PRINT_EVERY = 100          # print the loss every x iterations
SAVE_EVERY = 1000          # save the checkpoint every x iterations
