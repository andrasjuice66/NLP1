import numpy as np
import pandas as pd
from collections import Counter
from itertools import islice, tee

# Helper functions
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    result = []
    for i in range(len(iterable) - 1):
        result.append((iterable[i], iterable[i+1]))
    return result

def calculate_pmi(word_count, pair_count, total_count, pair):
    # Calculate the probability of each word and each pair of words
    w1, w2 = pair
    prob_w1 = word_count[w1] / total_count
    prob_w2 = word_count[w2] / total_count
    prob_pair = pair_count[pair] / total_count
    # Calculate PMI
    pmi = np.log2(prob_pair / (prob_w1 * prob_w2))
    return pmi

# Load and preprocess the corpus
with open('brown_100.txt', 'r') as file:
    # Read all words from the corpus, split by whitespace and convert to lowercase
    corpus = [word.lower() for line in file for word in line.split()]

# Count the frequency of each word and each pair of successive words
word_count = Counter(corpus)
pair_count = Counter(pairwise(corpus))
total_count = len(corpus)

# Filter out pairs that occur less than 10 times
pair_count = {pair: count for pair, count in pair_count.items() if count >= 10}

# Calculate PMI for each word pair
pmi_values = {pair: calculate_pmi(word_count, pair_count, total_count, pair) for pair in pair_count}

# Sort word pairs by PMI value
sorted_pmi = sorted(pmi_values.items(), key=lambda item: item[1], reverse=True)

# Extract the top 20 and bottom 20 word pairs by PMI
top_20_pmi = sorted_pmi[:20]
bottom_20_pmi = sorted_pmi[-20:]

# Output the results
print("Top 20 word pairs by PMI:")
for pair, pmi in top_20_pmi:
    print(f"{pair}: {pmi}")

print("\nBottom 20 word pairs by PMI:")
for pair, pmi in bottom_20_pmi:
    print(f"{pair}: {pmi}")

# Save PMI values to file
with open('unigram_pmi.txt', 'w') as f:
    for pair, pmi in sorted_pmi:
        f.write(f"{pair}: {pmi}\n")

