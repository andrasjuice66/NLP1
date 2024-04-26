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
from sklearn.preprocessing import normalize

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



print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))

#TODO: initialize numpy 0s array
f = codecs.open("brown_100.txt")
counts = np.zeros((len(word_index_dict), len(word_index_dict)), dtype=int)

previous_word = '<s>'
for line in f:
    words = line.lower().split()
    for word in words:
        if previous_word in word_index_dict and word in word_index_dict:
            counts[word_index_dict[previous_word]][word_index_dict[word]] += 1
        previous_word = word
    previous_word = '<s>' 

#print(len(word_index_dict))
#print(len(counts))

counts = counts.astype(float)
counts += 0.1


#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)


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




