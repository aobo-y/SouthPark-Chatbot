"""
SouthPark Chatbot
"""

import os
import argparse
import torch
from torch import optim

import config
from data_util import trimRareWords, load_pairs
from search_decoder import GreedySearchDecoder, BeamSearchDecoder
from seq_encoder import EncoderRNN
from seq_decoder_persona import DecoderRNN
from seq2seq import trainIters
from evaluate import evaluateInput
from embedding_map import EmbeddingMap

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def init_word_embedding(embedding_paths):
    print('Init word embedding from: ', ', '.join(embedding_paths))

    lines = []
    for embedding_path in embedding_paths:
        embedding_path = os.path.join(DIR_PATH, embedding_path)

        with open(embedding_path, encoding='utf-8') as file:
            lines += file.read().strip().split('\n')

    tokens_of_lines = [l.strip().split(' ') for l in lines]
    words = [l[0] for l in tokens_of_lines]
    embedding_of_words = [[float(str_emb) for str_emb in l[1:]] for l in tokens_of_lines]

    word_map = EmbeddingMap(words)
    print(f'Load {word_map.size()} word embedding')

    for special_token in config.SPECIAL_WORD_EMBEDDING_TOKENS.values():
        if not word_map.has(special_token):
            word_map.append(special_token)
            # also init the embedding for special token
            embedding_len = len(embedding_of_words[0])
            embedding_of_words.append([0] * embedding_len)

    weight = torch.FloatTensor(embedding_of_words)
    embedding = torch.nn.Embedding.from_pretrained(weight, False)

    return word_map, embedding

def init_persona_embedding(persons, size):
    person_map = EmbeddingMap(persons)
    person_map.append(config.NONE_PERSONA)

    # Initialize persona embedding with 0
    weight = torch.zeros((person_map.size(), size))
    embedding = torch.nn.Embedding.from_pretrained(weight, False)

    return person_map, embedding

def load_data(corpus_name, corpus_file, word_map):
    datafile = os.path.join('data', corpus_name, corpus_file)
    pairs = load_pairs(datafile)

    # Trim pairs with words not in embedding
    pairs = trimRareWords(word_map, pairs)
    return pairs



def build_model(checkpoint_name):
    checkpoint = None

    if checkpoint_name:
        load_filename = os.path.join(config.SAVE_DIR, config.MODEL_NAME, f'{config.ENCODER_N_LAYERS}-{config.DECODER_N_LAYERS}_{config.HIDDEN_SIZE}', f'{checkpoint_name}.tar')

        print('Load checkpoint file:', load_filename)
        # If loading on same machine the model was trained on
        # checkpoint = torch.load(load_filename)

        # current_iteration = checkpoint['iteration']
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(load_filename, map_location=device)
        embedding_sd = checkpoint['embedding']
        persona_sd = checkpoint['persona']

        word_map = EmbeddingMap()
        word_map.__dict__ = checkpoint['word_map_dict']
        person_map = EmbeddingMap()
        person_map.__dict__ = checkpoint['person_map_dict']

        # Load word embeddings
        embedding = torch.nn.Embedding(word_map.size(), config.HIDDEN_SIZE)
        embedding.load_state_dict(embedding_sd)

        # Load persona embeddings
        personas = torch.nn.Embedding(person_map.size(), config.PERSONA_SIZE)
        personas.load_state_dict(persona_sd)

    else:
        # Initialize word embeddings
        word_map, embedding = init_word_embedding(config.WORD_EMBEDDING_FILES)

        # Initialize persona embedding
        person_map, personas = init_persona_embedding(config.PERSONS, config.PERSONA_SIZE)



    # make sure config is the same as init
    assert embedding.embedding_dim == config.HIDDEN_SIZE
    assert personas.embedding_dim == config.PERSONA_SIZE

    print('Building encoder and decoder ...')

    # Initialize encoder & decoder models
    encoder = EncoderRNN(embedding, config.ENCODER_N_LAYERS, config.ENCODER_DROPOUT_RATE, config.RNN_TYPE)
    decoder = DecoderRNN(config.ATTN_MODEL, embedding, personas, word_map.size(),
                         config.DECODER_N_LAYERS, config.DECODER_DROPOUT_RATE, config.USE_PERSONA, config.RNN_TYPE)

    if checkpoint:
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder, embedding, personas, word_map, person_map, checkpoint



def train(pairs, encoder, decoder, embedding, personas, word_map, person_map, checkpoint):
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.LR * config.DECODER_LR)

    iteration = 0

    if checkpoint:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
        iteration = checkpoint['iteration']

    # Run training iterations
    iteration += 1
    print(f'Starting Training from iteration {iteration}!')
    trainIters(word_map, person_map, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, personas, config.N_ITER, iteration)


def chat(encoder, decoder, word_map, speaker_id):
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    if config.BEAM_SEARCH_ON:
        searcher = BeamSearchDecoder(encoder, decoder)
    else:
        searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, word_map, speaker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'}, default='train', help="mode. if not specified, it's in the train mode")
    parser.add_argument('--speaker', default='<none>')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()

    if args.mode == 'train':
        encoder, decoder, embedding, personas, word_map, person_map, checkpoint = build_model(args.checkpoint)

        pairs = load_data(config.CORPUS_NAME, config.CORPUS_FILE, word_map)

        train(pairs, encoder, decoder, embedding, personas, word_map, person_map, checkpoint)
    elif args.mode == 'chat':
        encoder, decoder, embedding, personas, word_map, person_map, _ = build_model(args.checkpoint)

        speaker_name = args.speaker
        if person_map.has(speaker_name):
            print('Selected speaker:', speaker_name)
            speaker_id = person_map.get_index(speaker_name)
            chat(encoder, decoder, word_map, speaker_id)
        else:
            print('Invalid speaker. Possible speakers:', person_map.tokens)

if __name__ =='__main__':
    main()
