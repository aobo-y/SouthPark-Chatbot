"""
SouthPark Chatbot
"""

import os
import argparse
import torch

import config
from utils import CheckpointManager
from data_util import trim_unk_data, load_pairs
from search_decoder import GreedySearchDecoder, BeamSearchDecoder
from models import Seq2Seq
from trainer import Trainer
from evaluate import evaluateInput
from embedding_map import EmbeddingMap

DIR_PATH = os.path.dirname(__file__)
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

SAVE_PATH = os.path.join(DIR_PATH, config.SAVE_DIR, config.MODEL_NAME)

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

    return word_map, weight

def init_persona_embedding(persons):
    person_map = EmbeddingMap(persons)
    person_map.append(config.NONE_PERSONA)

    return person_map

def load_data(corpus_path, word_map, persona_map, trim=True):
    datafile = os.path.join(DIR_PATH, corpus_path)
    pairs = load_pairs(datafile)

    # Trim pairs with words not in embedding
    if trim:
        pairs = trim_unk_data(pairs, word_map, persona_map)

    return pairs

def build_model(checkpoint):
    if checkpoint:
        word_map = checkpoint['word_map']
        person_map = checkpoint['person_map']

    else:
        # Initialize word embeddings
        word_map, pre_we_weight = init_word_embedding(config.WORD_EMBEDDING_FILES)

        person_map = init_persona_embedding(config.PERSONS)

    print(f'word embedding size {config.WORD_EMBEDDING_SIZE}, persona embedding size {config.PERSONA_EMBEDDING_SIZE}, hidden size {config.HIDDEN_SIZE}, layers {config.MODEL_LAYERS}')

    word_ebd_shape = (word_map.size(), config.WORD_EMBEDDING_SIZE)
    persona_ebd_shape = (person_map.size(), config.PERSONA_EMBEDDING_SIZE)

    model = Seq2Seq(word_ebd_shape, persona_ebd_shape, config.HIDDEN_SIZE, config.MODEL_LAYERS, config.MODEL_DROPOUT_RATE, config.RNN_TYPE, config.ATTN_TYPE)

    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_pretrained_word_ebd(pre_we_weight)

    # Use appropriate device
    model = model.to(device)

    return model, word_map, person_map



def train(mode, model, word_map, person_map, checkpoint, checkpoint_mng):
    trainer = Trainer(model, word_map, person_map, checkpoint_mng)

    if checkpoint:
        trainer.load(checkpoint)
    else:
        checkpoint_mng.save_meta(word_map=word_map, person_map=person_map)

    if mode == 'pretrain':
        corpus = config.PRETRAIN_CORPUS
        trim_corpus = True
        train_persona = False
        n_iter = config.PRETRAIN_N_ITER

    elif mode == 'finetune':
        # finetune requires checkpoint
        assert checkpoint is not None

        corpus = config.FINETUNE_CORPUS
        trim_corpus = False
        train_persona = True
        n_iter = config.FINETUNE_N_ITER

        # finetune from pretrain checkpoint, reset start iter
        # verify stage keyword to make it compatible with previous
        if 'stage' not in checkpoint or checkpoint['stage'] != 'finetune':
            trainer.reset_iter()

    pairs = load_data(corpus, word_map, person_map, trim_corpus)

    # Ensure dropout layers are in train mode
    model.train()

    # in pretrain stage, do not update personas
    if not train_persona:
        model.freeze_persona()

    trainer.train(pairs, n_iter, config.BATCH_SIZE, stage=mode)


def chat(model, word_map, speaker_id):
    # Set dropout layers to eval mode
    model.eval()

    # Initialize search module
    if config.BEAM_SEARCH_ON:
        searcher = BeamSearchDecoder(model)
    else:
        searcher = GreedySearchDecoder(model)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, word_map, speaker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices={'pretrain', 'finetune', 'chat'}, help="mode to run the chatbot")
    parser.add_argument('-s', '--speaker', default='<none>')
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()

    print('Saving path:', SAVE_PATH)
    checkpoint_mng = CheckpointManager(SAVE_PATH)

    checkpoint = None
    if args.checkpoint:
        print('Load checkpoint:', args.checkpoint)
        checkpoint = checkpoint_mng.load(args.checkpoint, device)

    model, word_map, person_map = build_model(checkpoint)

    if args.mode == 'pretrain' or args.mode == 'finetune':
        train(args.mode, model, word_map, person_map, checkpoint, checkpoint_mng)

    elif args.mode == 'chat':
        speaker_name = args.speaker
        if person_map.has(speaker_name):
            print('Selected speaker:', speaker_name)
            speaker_id = person_map.get_index(speaker_name)
            chat(model, word_map, speaker_id)
        else:
            print('Invalid speaker. Possible speakers:', person_map.tokens)

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--checkpoint')
    args = parser.parse_args()
    config.USE_PERSONA = True

    checkpoint_mng = CheckpointManager(SAVE_PATH)
    checkpoint = None if not args.checkpoint else checkpoint_mng.load(args.checkpoint, device)

    model, word_map, person_map = build_model(checkpoint)
    # Set dropout layers to eval mode
    model.eval()
    # Initialize search module
    if config.BEAM_SEARCH_ON:
        searcher = BeamSearchDecoder(model)
    else:
        searcher = GreedySearchDecoder(model)
    return searcher, word_map, person_map


if __name__ == '__main__':
    main()
