# coding=utf-8

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


def cal_bleu(hypothesis, \
            references, \
            n_gram=4, \
            individual_or_cumulative='cumulative', \
            smoothing_function=None):
    '''
    smoothing_function==chencherry.method1 or None
    '''
    # smooth function
    if smoothing_function=='method1':
        cc = SmoothingFunction()
        smoothing_function=cc.method1
    # get weight
    weight = decide_which_bleu(hypothesis, n_gram, individual_or_cumulative)
    # calculate bleu score
    if weight != -2 and weight != -1 and weight != 0:
        score = sentence_bleu(references, hypothesis, weights=weight, smoothing_function=smoothing_function)
    else:
        return -1

    return score


def decide_which_bleu(hypothesis, \
                      n_gram=4, \
                      individual_or_cumulative='cumulative'):
    '''
    individual_or_cumulative==['cumulative','individual'] chose one of two choices.
    '''
    if len(hypothesis)==0:
        print('error, length of hypothesis is zero')
        return -2
    if n_gram > 4:
        print('error, we think n=4 is enough, chose n_gram<=4')
        return -1

    if len(hypothesis)<n_gram:
        print('error, length of hypothesis small than n_gram')
        #n_gram = len(hypothesis)
        return 0

    weights1 = [(1,0,0,0), (0.5,0.5,0,0), (1.0/3,1.0/3,1.0/3,0), (0.25,0.25,0.25,0.25)]
    weights2 = [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)]
    if individual_or_cumulative=='cumulative':
        return weights1[n_gram-1]
    elif individual_or_cumulative=='individual':
        return weights2[n_gram-1]


if __name__ == "__main__":
    print('haha')
