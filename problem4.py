#!/usr/bin/env python
# coding: utf-8

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import codecs
import random

# load the indices dictionary
# implement code from part 1
vocab = codecs.open("brown_vocab_100.txt")
word_index_dict = {}
for i, line in enumerate(vocab):
    word_index_dict[line.rstrip()] = i

# TODO: write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as wf:
        # Convert dictionary to string and write to the file
        for word, idx in word_index_dict.items():
            wf.write(f'{word}: {idx}\n')



#TODO: initialize numpy 0s array
f = codecs.open("brown_100.txt")
counts = np.zeros((len(word_index_dict), len(word_index_dict)), dtype=int)

for line in f:
    previous_word = '<s>'
    words = line.lower().split()
    for word in words:
        if word == "<s>":
            continue
        if previous_word in word_index_dict and word in word_index_dict:
            counts[word_index_dict[previous_word]][word_index_dict[word]] += 1
        previous_word = word

probs = normalize(counts, norm='l1', axis=1)
counts = counts.astype(float)
# ADD SMOOTHING
counts += 0.1
smoothed_counts =  counts
smoothed_probs = normalize(smoothed_counts, norm='l1', axis=1)

#TODO: normalize counts
# Function to get the smoothed probability of a bigram
def get_smooth_probability(w1, w2):
    return probs[word_index_dict[w1]][word_index_dict[w2]]

# Writing specified probabilities to a file using smoothed probabilities
with codecs.open('smooth_probs.txt', 'w', encoding='utf-8') as out_file:
    bigrams = [('all', 'the'), ('the', 'jury'), ('the', 'campaign'), ('anonymous', 'calls')]
    for (prev, next) in bigrams:
        probability = get_smooth_probability(prev, next)
        out_file.write(f'p({next} | {prev}): {probability:.7f}\n')
f.close()




#TASK4.6
# Calculate and write out smoothed perplexities
with codecs.open("toy_corpus.txt", "r", encoding='utf-8') as corpus_file, codecs.open("smoothed_eval.txt", "w", encoding='utf-8') as output_file:
    for line in corpus_file:
        words = line.lower().strip().split()
        sentence_probability = 1.0
        bigram_count = len(words) - 1 
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            if current_word in word_index_dict and next_word in word_index_dict:
                current_index = word_index_dict[current_word]
                next_index = word_index_dict[next_word]
                sentence_probability *= smoothed_probs[current_index][next_index]
            else:
                sentence_probability *= 0  
        if sentence_probability > 0:
            perplexity = pow(sentence_probability, -1.0 / bigram_count)
            print(perplexity)

        else:
            perplexity = float('inf') 

        output_file.write(f'{perplexity}\n')

#TASK 4.7
with open('smoothed_generation.txt', 'w') as output_file:
    for i in range(10):
        string = GENERATE(word_index_dict, probs, 'bigram', 10, '<s>')
        output_file.write(f'{string}\n')