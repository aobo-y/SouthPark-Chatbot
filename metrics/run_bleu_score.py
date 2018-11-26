from six import text_type
from nltk.tokenize import word_tokenize
import string
import json

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

def run(configs):
    #
    type_seq2seq = configs['model']['type']
    val_or_test = configs['test_type']
    # read file
    with open(configs['file_dir'][type_seq2seq][val_or_test], 'w') as re:
        count_of_sent = 0
        sum_score = 0
        for line in re:
            # split input output and maybe person label
            collect = line.split('\t')
            input_sent = collect[0]
            reference_sent = collect[1]
            if type_seq2seq == 'personal':
                personal_label = collect[2]
            # get model output
            #TODO
            # calculate bleu score for this sentence
            score = cal_bleu(hypothesis, [reference], n_gram, individual_or_cumulative, smoothing_function)
            if score != -1:
                sum_score = sum_score + score
                count_of_sent += 1

            average_bleu = sum_score / count_of_sent
            return average_bleu

if __name__ == '__main__':
    configs = read_config('config.json')
    run(configs)
