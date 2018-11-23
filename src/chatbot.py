"""
SouthPark Chatbot
"""

import torch
import os
import argparse

from torch import nn
from torch import optim

import config
from data_util import loadPrepareData, trimRareWords
from search_decoder import GreedySearchDecoder, BeamSearchDecoder
from seq_encoder import EncoderRNN
from seq_decoder import LuongAttnDecoderRNN
from seq2seq import trainIters
from evaluate import evaluateInput

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def load_data(corpus_name=config.CORPUS_NAME, corpus_file=config.CORPUS_NAME):
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, config.CORPUS_FILE)
    # Load/Assemble voc and pairs
    voc, pairs = loadPrepareData(corpus, corpus_file, datafile)
    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, config.MIN_COUNT)
    return voc, pairs


def build_model(voc, load_checkpoint=config.LOAD_CHECKPOINT):
    if load_checkpoint:
        loadFilename = os.path.join(config.SAVE_DIR, config.MODEL_NAME, config.CORPUS_NAME,
                                '{}-{}_{}'.format(config.ENCODER_N_LAYERS, config.DECODER_N_LAYERS, config.HIDDEN_SIZE),
                                '{}_checkpoint.tar'.format(config.CHECKPOINT_ITER))
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
    else:
        loadFilename = None


    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, config.HIDDEN_SIZE)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(config.HIDDEN_SIZE, embedding, config.ENCODER_N_LAYERS, config.ENCODER_DROPOUT_RATE)
    decoder = LuongAttnDecoderRNN(config.ATTN_MODEL, embedding, config.HIDDEN_SIZE, voc.num_words, config.DECODER_N_LAYERS, config.DECODER_DROPOUT_RATE)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    if load_checkpoint:
      return encoder, decoder, loadFilename, encoder_optimizer_sd, decoder_optimizer_sd, embedding
    else:
      return encoder, decoder, loadFilename, None, None, embedding



def train(encoder, decoder, loadFilename, encoder_optimizer_sd, decoder_optimizer_sd, embedding):
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.LR * config.DECODER_LR)

    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    trainIters(config.MODEL_NAME, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, config.ENCODER_N_LAYERS, config.DECODER_N_LAYERS, config.HIDDEN_SIZE, config.SAVE_DIR,
               config.N_ITER, config.BATCH_SIZE, config.PRINT_EVERY, config.SAVE_EVERY, config.CLIP,
               config.TEACHER_FORCING_RATIO, config.CORPUS_NAME, loadFilename)


def chat(encoder, decoder, voc):
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = BeamSearchDecoder(encoder, decoder)
    # searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'}, default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()
    voc, pairs = load_data()

    if args.mode == 'train':
      encoder, decoder, loadFilename, encoder_optimizer_sd, decoder_optimizer_sd, embedding = build_model(voc)
      train(encoder, decoder, loadFilename, encoder_optimizer_sd, decoder_optimizer_sd, embedding)
    elif args.mode == 'chat':
      encoder, decoder, loadFilename, encoder_optimizer_sd, decoder_optimizer_sd, embedding = build_model(voc, load_checkpoint=True)
      chat(encoder, decoder, voc)
