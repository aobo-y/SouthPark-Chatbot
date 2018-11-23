"""
Evaluate
"""

import torch
from torch import tensor
import config
from data_util import normalizeString, indexesFromSentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=config.MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose dimensions of batch to match model's expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

# Evaluate inputs from user input (stdin)
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        # get input sentence
        input_sentence = input('> ')
        # check if it is quit case
        if input_sentence == 'q' or input_sentence =='quit':
            break
        input_sentence = normalizeString(input_sentence)
        # evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # format and print reponse sentence
        output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
        print('Bot:', ' '.join(output_words))
        '''
        try:
            # get input sentence
            input_sentence = input('> ')
            # check if it is quit case
            if input_sentence == 'q' or input_sentence =='quit':
                break
            input_sentence = normalizeString(input_sentence)
            # evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # format and print reponse sentence
            output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print('Error: Encountered unknown word.')
        '''


# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print('> '+sentence)
    # normalize sentence
    input_sentence = normalizeString(sentence)
    # evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
    print('Bot:', ' '.join(output_words))
