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

def run(config):
    #
    type_seq2seq = config['type']
    # read file
    r = open(config[])


if __name__ == '__main__':
    config = read_config('config.json')
    run(config)
