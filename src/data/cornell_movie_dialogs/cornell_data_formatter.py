# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
import csv
import os
import config


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs
  
if __name__=='__main__':
  corpus_name = config.CORPUS_NAME
  corpus = os.path.join("data", corpus_name)
  # Define path to new file
  datafile = os.path.join(corpus, "formatted_movie_lines.txt")
  delimiter = '\t'
  # Unescape the delimiter
  delimiter = str(codecs.decode(delimiter, "unicode_escape"))
  
  # Initialize lines dict, conversations list, and field ids
  lines = {}
  conversations = []
  MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
  MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
  
  # Load lines and process conversations
  print("\nProcessing corpus...")
  lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
  print("\nLoading conversations...")
  conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                    lines, MOVIE_CONVERSATIONS_FIELDS)
  
  # Write new csv file
  print("\nWriting newly formatted file...")
  with open(datafile, 'w', encoding='utf-8', newline='') as outputfile:
      writer = csv.writer(outputfile, delimiter=delimiter)
      for pair in extractSentencePairs(conversations):
          writer.writerow(pair)
