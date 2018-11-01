""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

This file contains the hyperparameters for the model.

See README.md for instruction on how to run the starter code.
"""

# parameters for processing the dataset
DATA_PATH = 'data/cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

MAX_ITER = 10000

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(8, 10), (12, 14), (16, 19)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
				("don ' t ", "do n't "), ("didn ' t ", "did n't "), ("doesn ' t ", "does n't "),
				("can ' t ", "ca n't "), ("shouldn ' t ", "should n't "), ("wouldn ' t ", "would n't "),
				("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3        # the number of GRU/LSTM layers
USE_LSTM = True       # use GRU or LSTM cell
EMBEDDING_SIZE = 256  # the length of the embedding vector for each symbol
ATTENTION_HEADS = 5
BATCH_SIZE = 64

LR = 0.01
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 24361
DEC_VOCAB = 24538
