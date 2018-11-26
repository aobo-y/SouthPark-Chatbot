from six import text_type
from nltk.tokenize import word_tokenize
import string

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


