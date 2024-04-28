#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
   with open("brown_vocab_100.txt", 'r') as file:
        for index, line in enumerate(file):
            word_index_dict[line.rstrip()] = index

# TODO: write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as wf:
        # Convert dictionary to string and write to the file
        for word, idx in word_index_dict.items():
            wf.write(f'{word}: {idx}\n')    

#TODO: iterate through file and update counts
f = open("brown_100.txt")
counts = np.zeros(len(word_index_dict))
with f as file:
    for line in file:
        words = line.lower().split()
        for word in words:
            if word in word_index_dict:
                counts[word_index_dict[word]] += 1
f.close()
#print(counts)

#TODO: normalize and writeout counts. 
probs = counts / np.sum(counts)
with open('unigram_probs.txt', 'w') as wf:
    for probability in probs:
        wf.write(f'{probability}\n')



# TASK 4.6
def calculate_sentence_probability(sentence, word_index_dict, unigram_probs):
    words = sentence.lower().split()
    sentence_prob = 1.0
    sent_len = len(words)
    for word in words:
        if word in word_index_dict:  # Check if the word is in the vocabulary
            sentence_prob *= unigram_probs[word_index_dict[word]]
        else:
            sentence_prob *= 0  # Assign zero probability if word is not in the vocabulary

    perplexity = 1/(pow(sentence_prob, 1.0/sent_len))
    return sentence_prob, perplexity 

# Read sentences from the toy corpus and calculate their probabilities
with open("toy_corpus.txt", 'r') as corpus_file, open('unigram_eval.txt', 'w') as output_file:
    for sentence in corpus_file:
        sentence = sentence.strip()  # Remove the newline character
        if sentence:  # Check if the sentence is not empty
            probability, perplexity = calculate_sentence_probability(sentence, word_index_dict, probs)
            output_file.write(f'{perplexity}\n')  # Write the probability to the output file

