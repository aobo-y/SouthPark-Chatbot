# -*- coding: utf-8 -*-

import re
import unicodedata
import itertools
import torch
import config

# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()

    # trim ending single stop
    s = re.sub(r'([^\.])(\.)$', r'\1', s)
    # give a leading & ending spaces to punctuations
    s = re.sub(r'([.!?,])', r' \1 ', s)
    # purge unrecognized token with space
    s = re.sub(r'[^0-9a-z.!?,]+', r' ', s)
    # squeeze multiple spaces
    s = re.sub(r'([ ]+)', r' ', s)
    # expand the abbreviation
    s = re.compile("n\'t").sub(' not', s)
    s = re.compile("\'d").sub(' would', s)
    s = re.compile("\'ll").sub(' will', s)
    s = re.compile("\'m").sub(' am', s)
    s = re.compile("\'re").sub(' are', s)
    s = re.compile("\'ve").sub(' have', s)
    # remove extra leading & ending space
    return s.strip()

def normalize_name(s):
    return s.lower()

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

def filter_pair(pair):
    '''
    Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    '''
    # TODO: data contains garbage, pretrain data has 2 tab split

    if len(pair) <= 3 and len(pair) >= 2:
        return len(pair[0].split(' ')) < config.MAX_LENGTH and len(pair[1].split(' ')) < config.MAX_LENGTH
    else:
        return False

# Using the functions defined above, return a populated voc object and pairs list
def load_pairs(datafile):
    print("Start preparing training data ...")

    print("Reading lines from %s..." % datafile)
    # Read the file and split into lines
    with open(datafile, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s for s in l.split('\t')] for l in lines]

    # normalize
    for pair in pairs:
        pair[0] = normalizeString(pair[0])
        pair[1] = normalizeString(pair[1])
        if len(pair) == 3:
            pair[2] = normalize_name(pair[2])

    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = [pair for pair in pairs if filter_pair(pair)]
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))

    return pairs


def trim_unk_data(pairs, voc, person_map):
    # Filter out pairs with trimmed words
    keep_pairs = []

    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]

        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if not voc.has(word):
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if not voc.has(word):
                keep_output = False
                break

        if len(pair) == 3:
            speaker = pair[2]
            if not person_map.has(speaker):
                keep_output = False

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexes_from_sentence(sentence, voc):
    unk = voc.unk
    eos = voc.eos

    tokens = [word if voc.has(word) else unk for word in sentence.split(' ')]
    tokens.append(eos)

    return [voc.get_index(token) for token in tokens]


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
def input_var(input_batch, padding):
    lengths = torch.tensor([len(indexes) for indexes in input_batch])
    pad_list = zeroPadding(input_batch, padding)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def output_var(indexes_batch, padding):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch, padding)
    mask = binaryMatrix(pad_list, padding)
    mask = torch.ByteTensor(mask)

    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch, padding):
    # sort by input length, no idea why
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)

    input_batch = [pair[0] for pair in pair_batch]
    output_batch = [pair[1] for pair in pair_batch]
    speaker_batch = [pair[2] for pair in pair_batch]

    inp, lengths = input_var(input_batch, padding)
    output, mask, max_target_len = output_var(output_batch, padding)

    # Return speaker_variable tensor with shape=(batch_size)
    speaker_variable = torch.LongTensor(speaker_batch)
    return inp, lengths, output, mask, max_target_len, speaker_variable


def data_2_indexes(pair, voc, persons):
    speaker = pair[2] if len(pair) == 3 else persons.none

    return [
        indexes_from_sentence(pair[0], voc),
        indexes_from_sentence(pair[1], voc),
        persons.get_index(speaker)
    ]
