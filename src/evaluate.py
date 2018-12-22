"""
Evaluate
"""

import re
import datetime
import torch
import config
from data_util import normalizeString, indexes_from_sentence


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(searcher, voc, sentence, speaker_id):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(sentence, voc)]
    # create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose dimensions of batch to match model's expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence with searcher
    sos = voc.sos_idx
    eos = voc.eos_idx
    tokens, scores = searcher(input_batch, lengths, speaker_id, sos, eos)
    # indexes -> words
    decoded_words = [voc.get_token(token.item()) for token in tokens]
    return decoded_words

# Evaluate inputs from user input (stdin)
def evaluateInput(searcher, voc, speaker_id):
    input_sentence = ''
    while 1:
        try:
            # get input sentence
            input_sentence = input('> ')
            # check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            input_sentence = normalizeString(input_sentence)
            # evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence, speaker_id)
            # format and print reponse sentence
            output_words = [x for x in output_words if x not in [voc.eos, voc.pad]]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print('Error: Encountered unknown word.')


# Normalize input sentence and call evaluate()
def evaluateExample(sentence, searcher, voc, speaker_id):
    print('> ' + sentence)
    # normalize sentence
    input_sentence = normalizeString(sentence)
    # evaluate sentence
    output_words = evaluate(searcher, voc, input_sentence, speaker_id)
    output_words = [x for x in output_words if x not in [voc.eos, voc.pad]]
    output_words[0] = output_words[0].capitalize()
    res = ' '.join(output_words)
    res = re.sub(r'\s+([.!?,])', r'\1', res)
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}> '.format(datetime.datetime.now()) + res)
    return res
