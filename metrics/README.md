# Evaluation Part

## 1. 

First run `python3 split_data.py` in src/data to get splited train/val/test data.

the data sets are stored in general_data directory and persona_data directory

because our chatbot has personality, so there are two kinds of data to evaluate.

you can see the readme in /src/data to get more details.

Then can use val/test data to calculate BLEU score.


## 2.

In python command, Run 

> import nltk

> nltk.download('punkt')

## 3.

Run `python3 run_bleu_score.py` to calculate BLEU score.
