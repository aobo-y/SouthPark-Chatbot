# -*- coding: utf-8 -*-

import torch
import re
import unicodedata
import itertools
import config


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_TOKEN: "PAD", config.SOS_TOKEN: "SOS", 
                           config.EOS_TOKEN: "EOS", config.UNK_TOKEN: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK
        self.num_people = 1
        self.people2index = {'NONE':0}

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
    
    def addPeople(self, word):
        if word not in self.people2index:
            self.people2index[word] = self.num_people
            self.num_people += 1
        

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


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p, use_persona=config.USE_PERSONA):
    # Input sequences need to preserve the last word for EOS token
    if len(p)==3 and use_persona:
        return len(p[0].split(' ')) < config.MAX_LENGTH and len(p[1].split(' ')) < config.MAX_LENGTH and len(p[2]) > 1
    elif len(p)==2 and use_persona is not True:
        return len(p[0].split(' ')) < config.MAX_LENGTH and len(p[1].split(' ')) < config.MAX_LENGTH
    else:
        return False


# Filter pairs using filterPair condition
def filterPairs(pairs):
    outs=[]
    for pair in pairs:
        if filterPair(pair, config.USE_PERSONA) :
            outs.append(pair)
    return outs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        if len(pair)==3 and config.USE_PERSONA:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
            voc.addPeople(pair[2])
        elif config.USE_PERSONA is not True:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    print("Counted people:", voc.num_people)
    return voc, pairs


def trimRareWords(voc, pairs, min_count=config.MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def indexesFromSentence(voc, sentence):
    words = []
    for word in sentence.split(' '):
        if word not in voc.word2index.keys():
            words.append(config.UNK_TOKEN)
        else:
            words.append(voc.word2index[word])
    return words + [config.EOS_TOKEN]


def zeroPadding(l, fillvalue=config.PAD_TOKEN):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=config.PAD_TOKEN):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == config.PAD_TOKEN:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Return speaker_ids tensor with shape=(max_length, 1, batch_size)
def speakerToId(speaker_batch, voc):
    speaker_ids = []
    for speaker in speaker_batch:
        speaker_id = voc.people2index[speaker]
        speaker_ids.append([speaker_id]*config.MAX_LENGTH)
    speaker_ids = torch.LongTensor(speaker_ids)
    speaker_ids = speaker_ids.t()
    speaker_ids = torch.unsqueeze(speaker_ids, 1)
    return speaker_ids
        
    
# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch, speaker_batch = [], [], []
    for pair in pair_batch:
        if len(pair)==3 and config.USE_PERSONA:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
            speaker_batch.append(pair[2])
        elif config.USE_PERSONA is not True:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
            speaker_batch.append('NONE')
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    speaker = speakerToId(speaker_batch, voc)
    return inp, lengths, output, mask, max_target_len, speaker
