"""
Evaluate
"""

import torch
import config
from data_util import normalizeString, indexes_from_sentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def evaluate(searcher, word_map, sentence, speaker_id):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(word_map, sentence)]
    # create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # transpose dimensions of batch to match model's expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, speaker_id)
    # indexes -> words
    decoded_words = [word_map.get_token[token.item()] for token in tokens]
    return decoded_words

# Evaluate inputs from user input (stdin)
def evaluateInput(searcher, word_map, speaker_id):
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
            output_words = evaluate(searcher, word_map, input_sentence, speaker_id)
            # format and print reponse sentence
            output_words = [x for x in output_words if x not in config.SPECIAL_WORD_EMBEDDING_TOKENS.keys()]
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print('Error: Encountered unknown word.')


# Normalize input sentence and call evaluate()
def evaluateExample(sentence, searcher, word_map, speaker_id):
    print('> ' + sentence)
    # normalize sentence
    input_sentence = normalizeString(sentence)
    # evaluate sentence
    output_words = evaluate(searcher, word_map, input_sentence, speaker_id)
    output_words = [x for x in output_words if x not in config.SPECIAL_WORD_EMBEDDING_TOKENS.keys()]

    print('Bot:', ' '.join(output_words))
