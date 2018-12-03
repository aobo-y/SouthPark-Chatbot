#! /usr/bin/env python3
# -*- coding: utf-8
#
# @File:      sp_data_formatter
# @Brief:     Format south park data
# @Created:   Nov/21/2018
# @Author:    Jiahao Cai
#

import pickle
import heapq
import re

with open('south_park.pickle', 'rb') as f:
  # DATA = [{'scenario', 'character', 'line'}]
  DATA = pickle.load(f)

class char_freq:
  def __init__(self, name, freq):
    self.name = name
    self.freq = freq
  def __lt__(self, other):
    return self.freq < other.freq
  def __str__(self):
    return "%s: %d" % (self.name, self.freq)

def preprocess():
  for d in DATA:
    d['line'] = re.sub(r'\[.*?\]', '', d['line']) # trim scenario, e.g. '[drives by and honks] Ahh!'
    d['line'] = re.sub(r'[\(\)]', '', d['line']) # trim parens, e.g. '(Don't worry, I'm alright. Argh!)'
    d['line'] = re.sub(r'\n', ' ', d['line']) # ger rid of newlines, e.g. '\nSpank it, ever so gently.\n\n'
    d['line'] = re.sub(r'\s+', ' ', d['line']) # merge multi-spaces into one, e.g. 'Spank  it, ever so gently'
    d['line'] = re.sub(r'^\s+', '', d['line']) # trim leading spaces, e.g. ' Spank it, ever so gently.'
    d['line'] = re.sub(r'\s+$', '', d['line']) # trim trailing spaces, e.g. 'Spank it, ever so gently.  '
  

def get_topN_talker(N = 10):
  chars_record = {}
  for d in DATA:
    chars_record[d['character']] = chars_record.get(d['character'], 0) + 1
  heap = []
  for key in chars_record:
    heapq.heappush(heap, char_freq(key, chars_record[key]))
  heapq.heapify(heap)
  print("Here comes the top %d talkers: " % N)
  talkers = heapq.nlargest(N, heap)
  list(map(print, talkers))
  return list(map(lambda x: x.name, talkers))

def merge_scenario():
  scenario_set = {}
  for d in DATA:
    if scenario_set.get(d['scenario']) == None:
      scenario_set[d['scenario']] = [(d['line'], d['character'])] # (line, character)
    else:
      scenario_set[d['scenario']].append((d['line'], d['character'])) # (line, character)
  return scenario_set


SEPARATOR = '\t'

"""
line format for fine tune:
- ""\tKyle!\tCartman\n
Or
- You are fat.\tI'm not fat\tCartman\n
"""
FINE_TUNE_FORMAT = '%s' + SEPARATOR + '%s' + SEPARATOR + '%s\n'
def gen_fine_tune_file(talkers, scenario_set):
  print("Generating data for fine tune...")
  buf = ""
  for _, conversation in scenario_set.items():
    # item = (line, character)
    for i, item in enumerate(conversation):
      line = item[0]
      char = item[1]
      # The first line is from one of the top talkers
      if i == 0 and char in talkers: 
        buf += FINE_TUNE_FORMAT % ('\x00', line, char)
      # The answer is from one of the top talkers
      if i > 0 and char in talkers:
        last_line = conversation[i - 1][0]
        buf += FINE_TUNE_FORMAT % (last_line, line, char)
  with open('fine_tune.txt', 'w', encoding='utf8') as f:
    f.write(buf)
  print("Done!")


"""
line format for general train:
- ""\tDamn it!\n
Or
- How are you?\tGood.\n
"""
GENERAL_TRAIN_FORMAT = '%s' + SEPARATOR + '%s' + '\n'
def gen_general_train_file(talkers, scenario_set):
  print("Generating data for general training...")
  buf = ""
  for _, conversation in scenario_set.items():
    # item = (line, character)
    for i, item in enumerate(conversation):
      line = item[0]
      char = item[1]
      # The first line is NOT from one of the top talkers
      if i == 0 and char not in talkers: 
        buf += GENERAL_TRAIN_FORMAT % ('\x00', line)
      # The answer is NOT from one of the top talkers
      if i > 0 and char not in talkers:
        last_line = conversation[i - 1][0]
        buf += GENERAL_TRAIN_FORMAT % (last_line, line)
  with open('general_train.txt', 'w', encoding='utf8') as f:
    f.write(buf)
  print("Done!")



if __name__ == "__main__":
  preprocess()
  talkers = get_topN_talker()
  scenario_set = merge_scenario()
  gen_fine_tune_file(talkers, scenario_set)
  gen_general_train_file(talkers, scenario_set)

