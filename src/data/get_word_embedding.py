'''
download the word embedding and filter it based on our corpus
'''

import urllib.request
import os
import shutil
import zipfile
import re
import numpy as np


DIR_PATH = os.path.dirname(__file__)

GLOVE_WE_URL = 'http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip'

WE_FOLDER = os.path.join(DIR_PATH, 'word_embedding')
ZIP_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.zip')
WE_FILE = os.path.join(WE_FOLDER, 'glove.42B.300d.txt')
# WE_FILE = os.path.join(WE_FOLDER, 'glove.6B.300d.txt')

CORPUS_FILES = [
    os.path.join(DIR_PATH, 'cornell_movie_dialogs/formatted_movie_lines.txt'),
    os.path.join(DIR_PATH, 'south_park/general_train.txt'),
    os.path.join(DIR_PATH, 'south_park/fine_tune.txt'),
]

OUTPUT_FILES = [
    os.path.join(WE_FOLDER, 'filtered.glove.42B.300d.part1.txt'),
    os.path.join(WE_FOLDER, 'filtered.glove.42B.300d.part2.txt')
]

# same as data util, better import from somewhere
def normalizeString(s):
    s = s.lower()
    # give a leading & ending spaces to punctuations
    s = re.sub(r'([.!?,])', r' \1 ', s)
    # purge unrecognized token with space
    s = re.sub(r'[^a-z.!?,]+', r' ', s)
    # squeeze multiple spaces
    s = re.sub(r'([ ]+)', r' ', s)
    # remove extra leading & ending space
    return s.strip()

def main():
    if not os.path.exists(WE_FOLDER):
        print('create folder', WE_FOLDER)
        os.mkdir(WE_FOLDER)

    if not os.path.exists(WE_FILE):
        print('download embedding from', GLOVE_WE_URL)
        # Download the file from `url` and save it locally under:
        with urllib.request.urlopen(GLOVE_WE_URL) as response, open(ZIP_FILE, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(WE_FOLDER)

        os.remove(ZIP_FILE)

    voc = {}

    for corpus_file in CORPUS_FILES:
        print('read corpus:', corpus_file)

        with open(corpus_file, 'r', encoding='utf8') as file:
            lines = file.read().strip().split('\n')
            for line in lines:
                pairs = line.split('\t')
                for sentence in pairs[:1]:
                    sentence = normalizeString(sentence)
                    if sentence == '':
                        continue

                    for token in sentence.split(' '):
                        if token not in voc:
                            voc[token] = 0
                        voc[token] += 1

    print('size of the data:', len(voc))

    filtered_line = []
    with open(WE_FILE, 'r', encoding='utf8') as file:
        # emdedding file is too large, read line by line
        size = 0
        line = file.readline()
        while line:
            size += 1

            token = line[: line.find(' ')]
            if token in voc or token == '<unk>':
                filtered_line.append(line)

            line = file.readline()

        print('size of the embedding:', size)

    print('size of the filtered embedding:', len(filtered_line))

    # embedding is too big to commit
    splited_lines = np.array_split(filtered_line, len(OUTPUT_FILES))
    for lines, output_file in zip(splited_lines, OUTPUT_FILES):
        with open(output_file, 'w', encoding='utf8') as file:
            # remove last line's newline
            lines[-1] = lines[-1].rstrip('\n')
            file.write(''.join(lines))


if __name__ == '__main__':
    main()
