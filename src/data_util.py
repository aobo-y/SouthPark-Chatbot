# -*- coding: utf-8 -*-

import re
import unicodedata
import itertools
import torch
import config


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_TOKEN: "PAD", config.SOS_TOKEN: "SOS",
                           config.EOS_TOKEN: "EOS", config.UNK_TOKEN: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK
        self.num_people = 11
        self.people2index = {'NONE': 0, 'kyle': 1, 'cartman': 2, 'stan': 3,
                             'chef': 4, 'kenny': 5, 'mr . garrison': 6,
                             'randy': 7, 'sharon': 8, 'gerald': 9, 'butters': 10}

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_TOKEN: "PAD", config.SOS_TOKEN: "SOS", config.EOS_TOKEN: "EOS", config.UNK_TOKEN: "UNK"}
        self.num_words = 3 # Count default tokens
        for word in keep_words:
            self.addWord(word)


# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Read query/response pairs
def readVocs(datafile):
    print("Reading lines from %s..." % datafile)
    # Read the file and split into lines
    with open(datafile, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    return pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    if len(p) == 3 and config.USE_PERSONA:
        return len(p[0].split(' ')) < config.MAX_LENGTH and len(p[1].split(' ')) < config.MAX_LENGTH and len(p[2]) > 1
    elif len(p) == 2 and config.USE_PERSONA is not True:
        return len(p[0].split(' ')) < config.MAX_LENGTH and len(p[1].split(' ')) < config.MAX_LENGTH
    else:
        return False

# Using the functions defined above, return a populated voc object and pairs list
def load_pairs(datafile):
    print("Start preparing training data ...")

    print("Reading lines from %s..." % datafile)
    # Read the file and split into lines
    with open(datafile, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = [pair for pair in pairs if filterPair(pair)]
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    return pairs


def trimRareWords(word_map, pairs):
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word_map.has(word):
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word_map.has(word):
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexes_from_sentence(word_map, sentence):
    unk = config.SPECIAL_WORD_EMBEDDING_TOKENS['UNK']
    eos = config.SPECIAL_WORD_EMBEDDING_TOKENS['EOS']

    tokens = [word if word_map.has(word) else unk for word in sentence.split(' ')]
    tokens.append(eos)

    return [word_map.get_index(token) for token in tokens]


def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, fillvalue):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for index in seq:
            if index == fillvalue:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def input_var(l, word_map):
    pad = config.SPECIAL_WORD_EMBEDDING_TOKENS['PAD']
    fillvalue = word_map.get_index(pad)

    indexes_batch = [indexes_from_sentence(word_map, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch, fillvalue)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_var(l, word_map):
    pad = config.SPECIAL_WORD_EMBEDDING_TOKENS['PAD']
    fillvalue = word_map.get_index(pad)

    indexes_batch = [indexes_from_sentence(word_map, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch, fillvalue)
    mask = binaryMatrix(pad_list, fillvalue)
    mask = torch.ByteTensor(mask)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# Return speaker_ids tensor with shape=(max_length, 1, batch_size)
def speaker_var(speaker_batch, person_map):
    indexes_batch = [person_map.get_index(speaker) for speaker in speaker_batch]

    return torch.LongTensor([indexes_batch]) # decoder need length as first dimension


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch, word_map, person_map):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch, speaker_batch = [], [], []

    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

        if len(pair) == 3 and config.USE_PERSONA:
            speaker_batch.append(pair[2])
        elif config.USE_PERSONA is not True:
            speaker_batch.append(config.NONE_PERSONA)

    inp, lengths = input_var(input_batch, word_map)
    output, mask, max_target_len = output_var(output_batch, word_map)

    speaker_input = speaker_var(speaker_batch, person_map)
    return inp, lengths, output, mask, max_target_len, speaker_input
