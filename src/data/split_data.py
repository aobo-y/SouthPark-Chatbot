# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:45:17 2018

@author: Wanyu Du
"""

import random
import os

def split_train_test_val(input_file, split_ratio, use_shuffle):
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
    print('Read {!s} sentences from {!s}.'.format(len(lines), input_file))
    train_len = int(len(lines)*(1-2*split_ratio))
    val_len = int(len(lines)*split_ratio)
    if use_shuffle:
        random.shuffle(lines)
    train_data = lines[:train_len]
    val_data = lines[train_len:train_len+val_len]
    test_data = lines[train_len+val_len:]
    return train_data, val_data, test_data
        

if __name__=='__main__':
    # generate cornell data
    cornell_train, cornell_val, cornell_test = split_train_test_val('cornell movie-dialogs corpus/formatted_movie_lines.txt',
                                                                    0.1, True)
    
    # generate south_park general data
    sp_general_train, sp_general_val, sp_general_test = split_train_test_val('south_park/general_train.txt', 0.1, True)
    
    # generate south_park fine tune data
    sp_person_train, sp_person_val, sp_person_test = split_train_test_val('south_park/fine_tune.txt', 0.1, True)
    
    # generate simpsons fine tune data
    simp_person_train, simp_person_val, simp_person_test = split_train_test_val('simpsons/fine_tune.txt', 0.1, True)
    
    # combine general data
    if not os.path.exists('general_data'):
        os.mkdir('general_data')
    general_train = cornell_train+sp_general_train
    general_val = cornell_val+sp_general_val
    general_test = cornell_test+sp_general_test
    with open('general_data/train.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into general_train.'.format(len(general_train)))
        f.writelines(general_train)
    with open('general_data/val.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into general_val.'.format(len(general_val)))
        f.writelines(general_val)
    with open('general_data/test.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into general_test.'.format(len(general_test)))
        f.writelines(general_test)
        
    # combine persona data
    if not os.path.exists('persona_data'):
        os.mkdir('persona_data')
    persona_train = sp_person_train+simp_person_train
    persona_val = sp_person_val+simp_person_val
    persona_test = sp_person_test+simp_person_test
    with open('persona_data/train.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into persona_train.'.format(len(persona_train)))
        f.writelines(persona_train)
    with open('persona_data/val.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into persona_val.'.format(len(persona_val)))
        f.writelines(persona_val)
    with open('persona_data/test.txt', 'w', encoding='utf8') as f:
        print('Write {!s} sentences into persona_test.'.format(len(persona_test)))
        f.writelines(persona_test)