from six import text_type
from nltk.tokenize import word_tokenize
import string
import json
import sys
import os
sys.path.insert(0, '../src/')
import chatbot
from chatbot import load_checkpoint
import bleu
import evaluate
from search_decoder import GreedySearchDecoder, BeamSearchDecoder

if not os.path.exists('checkpoints'):
    os.symlink('../src/checkpoints', 'checkpoints')
if not os.path.exists('data'):
    os.symlink('../src/data', 'data')


def make_tokens(sentence):
    # tokenize
    tokens = word_tokenize(sentence)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove puctuations
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    return words

def read_config(file_path):
    json_ob = json.load(open(file_path,'r'))
    return json_ob

def load_model(configs):
    checkpoint = load_checkpoint(configs['checkpointname'])
    #voc, pairs, encoder, decoder, load_filename, encoder_optimizer_sd, decoder_optimizer_sd, embedding, personas = chatbot.build_model(load_checkpoint=True)
    encoder, decoder, embedding, personas, word_map, person_map, checkpoint = chatbot.build_model(checkpoint)
    if configs['model']['search_method']=='greedy':
        searcher = GreedySearchDecoder(encoder, decoder)
    elif configs['model']['search_method']=='beam':
        searcher = BeamSearchDecoder(encoder, decoder)

    #speaker = voc.people2index[configs['model']['speaker_name']]
    return searcher, word_map, person_map

def run(configs):
    # configs
    type_seq2seq = configs['model']['type']
    val_or_test = configs['test_type']
    n_gram = configs['bleu']['n_gram']
    individual_or_cumulative = configs['bleu']['individual_or_cumulative']
    smoothing_function = configs['bleu']['smoothing_function']
    if smoothing_function == "None":
        smoothing_function = None
    # load model
    searcher, word_map, person_map = load_model(configs)
    speaker_id = person_map.get_index(configs['model']['speaker_name'])
    # read file
    file_path = configs['file_dir'][type_seq2seq][val_or_test]
    #print(file_path)
    maxscore = 0
    with open(file_path, 'r') as re:
        count_of_sent = 0
        sum_score = 0
        for line in re:
            # split input output and maybe person label
            collect = line.split('\t')
            input_sent = collect[0]
            reference_sent = collect[1]
            if type_seq2seq == 'personal':
                personal_label = str.casefold(collect[2]).strip('\n')
                if str(personal_label) != configs['model']['speaker_name']:
                    continue
            # get model output
            out = evaluate.evaluate(searcher, word_map, input_sent, speaker_id)
            out[:] = [x for x in out if not (x=='EOS' or x=='PAD')]
            # make tokens
            hypothesis = make_tokens(' '.join(out))
            #print(hypothesis)
            reference = make_tokens(reference_sent)
            # calculate bleu score for this sentence
            score = bleu.cal_bleu(hypothesis, [reference], n_gram, individual_or_cumulative, smoothing_function)
            if score == 0:
                continue
            if score > maxscore:
                maxscore = score
            print(score)
            print(count_of_sent)
            if score != -1:
                sum_score = sum_score + score
                count_of_sent += 1

        average_bleu = sum_score / count_of_sent
        return average_bleu, maxscore

if __name__ == '__main__':
    configs = read_config('config.json')
    average_bleu, maxscore = run(configs)
    print('average_bleu:', average_bleu)
    print('max_bleu:', maxscore)
