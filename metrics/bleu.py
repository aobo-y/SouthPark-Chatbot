# coding=utf-8

import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def cal_bleu(hypothesis, references, weights, n_gram, individual_or_cumulative):


def decide_which_bleu(hypothesis, n_gram=4,individual_or_cumulative='cumulative'):
    if len(hypothesis)<n_gram:
        print('error, length of hypothesis small than n_gram')
        return 0

    weights1 = [(1,0,0,0), (0.5,0.5,0,0), (1.0/3,1.0/3,1.0/3,0), (0.25,0.25,0.25,0.25)]
    if individual_or_cumulative=='cumulative':
        if len(hypothesis)>=4:
            return (1.0/4, 1.0/4, 1.0/4, 1.0/4)
