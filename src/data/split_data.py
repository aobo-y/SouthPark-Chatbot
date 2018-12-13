# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 09:45:17 2018

@author: Wanyu Du
"""

import random
import os

def split_general_data(input_file, split_ratio, use_shuffle):
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    print(f'Read {len(lines)} sentences from {input_file}.')
    train_len = int(len(lines)*(1-2*split_ratio))
    val_len = int(len(lines)*split_ratio)
    if use_shuffle:
        random.shuffle(lines)
    train_data = lines[:train_len]
    val_data = lines[train_len:train_len+val_len]
    test_data = lines[train_len+val_len:]
    return train_data, val_data, test_data

def split_persona_data(input_file, split_ratio, use_shuffle):
    people = {'kyle': 1, 'cartman': 2, 'stan': 3,
              'chef': 4, 'kenny': 5, 'mr. garrison': 6,
              'randy': 7, 'sharon': 8, 'gerald': 9, 'butters': 10}
    with open(input_file, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    print(f'Read {len(lines)} sentences from {input_file}.')

    person_train = []
    person_val = []
    person_test = []
    for person in people.keys():
        person_sents = []
        for line in lines:
            items = line.split('\t')
            if items[2].lower() == person:
                person_sents.append(line)
        train_len = int(len(person_sents)*(1-2*split_ratio))
        val_len = int(len(person_sents)*split_ratio)
        person_train += person_sents[:train_len]
        person_val += person_sents[train_len:train_len+val_len]
        person_test += person_sents[train_len+val_len:]
    return person_train, person_val, person_test

def remove_character_label(sents_list):
    for i, line in enumerate(sents_list):
        items = line.split('\t')
        new_line = items[0] + '\t' + items[1]
        sents_list[i] = new_line
    return sents_list


if __name__=='__main__':
    split_ratio = 0.1
    use_shuffle = True
    
    # generate cornell data
    cornell_train, cornell_val, cornell_test = split_general_data('cornell_movie_dialogs/formatted_movie_lines.txt', split_ratio, use_shuffle)
    # generate south_park general data
    sp_general_train, sp_general_val, sp_general_test = split_general_data('south_park/general_train.txt', split_ratio, use_shuffle)
    # generate simpsons general data
    simp_general_train, simp_general_val, simp_general_test = split_general_data('simpsons/fine_tune.txt', split_ratio, use_shuffle)
    # generate south_park fine tune data
    sp_person_train, sp_person_val, sp_person_test = split_persona_data('south_park/fine_tune.txt', split_ratio, use_shuffle)
    
    # remove the unused character label in simpsons data
    simp_general_train = remove_character_label(simp_general_train)
    simp_general_val = remove_character_label(simp_general_val)
    simp_general_test = remove_character_label(simp_general_test)
    
    # combine general data
    if not os.path.exists('general_data'):
        os.mkdir('general_data')
    general_train = cornell_train + sp_general_train + simp_general_train
    general_val = cornell_val + sp_general_val + simp_general_val
    general_test = cornell_test + sp_general_test + simp_general_test
    with open('general_data/train.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(general_train)} sentences into general_train.')
        for l in general_train:
            f.write(l+'\n')
    with open('general_data/val.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(general_val)} sentences into general_val.')
        for l in general_val:
            f.write(l+'\n')
    with open('general_data/test.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(general_test)} sentences into general_test.')
        for l in general_test:
            f.write(l+'\n')

    # combine persona data
    if not os.path.exists('persona_data'):
        os.mkdir('persona_data')

    with open('persona_data/train.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(sp_person_train)} sentences into persona_train.')
        for l in sp_person_train:
            f.write(l+'\n')
    with open('persona_data/val.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(sp_person_val)} sentences into persona_val.')
        for l in sp_person_val:
            f.write(l+'\n')
    with open('persona_data/test.txt', 'w', encoding='utf8') as f:
        print(f'Write {len(sp_person_test)} sentences into persona_test.')
        for l in sp_person_test:
            f.write(l+'\n')