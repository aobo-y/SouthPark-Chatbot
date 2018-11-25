#! /usr/bin/env python3
# -*- coding: utf-8
#
# @File:      simpsons_data_formatter
# @Brief:     Format Simpsons data on Kaggle
# @Created:   Nov/24/2018
# @Author:    Jiahao Cai
#

import csv
import heapq
import re


with open('raw_simpsons_script_lines.csv', 'r') as f:
  # original data format
  # 0. id, 1. episode_id, 2. line number in episode, 3. raw_text,
  # 4. timestamp_in_ms, 5. speaking_line, 6. character_id, 7. location_id,
  # 8. raw_character_text, 9. raw_location_text, 10. spoken_words, 11. normalized_text,
  # 12. word_count
  # we need (0.id, 8.raw_character_text, 10.spoken_words)
  lines = list(csv.reader(f, delimiter = ',', quotechar = '"'))[1:] # the first line is not usable
  lines = sorted(lines, key = lambda x: int(x[0])) # sort lines by id
  # [(id, character, line)]
  DATA = []
  for line in lines:
    if line[5] == 'true': # if this is a speaking line
      DATA.append({'id':line[0], 'character': line[8], 'line':line[10]})

def preprocess():
  for d in DATA:
    d['line'] = re.sub(r'\[.*?\]', '', d['line']) # trim scenario, e.g. '[drives by and honks] Ahh!'
    d['line'] = re.sub(r'[\(\)]', '', d['line']) # trim parens, e.g. '(Don't worry, I'm alright. Argh!)'
    d['line'] = re.sub(r'\n', ' ', d['line']) # ger rid of newlines, e.g. '\nSpank it, ever so gently.\n\n'
    d['line'] = re.sub(r'\s+', ' ', d['line']) # merge multi-spaces into one, e.g. 'Spank  it, ever so gently'
    d['line'] = re.sub(r'^\s+', '', d['line']) # trim leading spaces, e.g. ' Spank it, ever so gently.'
    d['line'] = re.sub(r'\s+$', '', d['line']) # trim trailing spaces, e.g. 'Spank it, ever so gently.  '

SEPARATOR = '\t'

"""
line format for fine tune:
- ""\tKyle!\tCartman\n
Or
- You are fat.\tI'm not fat\tCartman\n
"""
FINE_TUNE_FORMAT = '%s' + SEPARATOR + '%s' + SEPARATOR + '%s\n'
def gen_fine_tune_file():
  print("Generating data for fine tune...")
  buf = ""
  for i, item in enumerate(DATA):
    # d = (id, character, line)
      char = item['character']
      line = item['line']
      if i == 0: 
        buf += FINE_TUNE_FORMAT % ('\x00', line, char)
      else:
        last_line = DATA[i - 1]['line']
        buf += FINE_TUNE_FORMAT % (last_line, line, char)
  with open('fine_tune.txt', 'w') as f:
    f.write(buf)
  print("Done!")



if __name__ == "__main__":
  preprocess()
  gen_fine_tune_file()

