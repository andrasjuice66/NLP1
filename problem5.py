from nltk import bigrams, trigrams, FreqDist

# Load the text data from the provided file and prepare to count bigrams and trigrams
file_path = 'brown_100.txt'

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Preprocessing: split the text into words
import re
words = re.findall(r'\b\w+\b', text.lower())

#print(words)
# Generate bigrams and trigrams
bigrams_list = list(bigrams(words))
trigrams_list = list(trigrams(words))
#print(bigrams_list, trigrams_list)

# Calculate frequency distributions
bigram_freq = FreqDist(bigrams_list)
trigram_freq = FreqDist(trigrams_list)

vocab_size = len(set(words))
#print(bigram_freq, trigram_freq, vocab_size)
print(bigram_freq)


def trigram_probability(w1, w2, w3, bigram_counts, trigram_counts, vocab_size, alpha, smoothed=False):
    bigram = (w1, w2)
    trigram = (w1, w2, w3)
    
    bigram_count = bigram_counts.get(bigram, 0)
    trigram_count = trigram_counts.get(trigram, 0)
    
    if smoothed:
        return (trigram_count + alpha) / (bigram_count + vocab_size*alpha)
    else:
        if bigram_count == 0:
            return 0  
        return trigram_count / bigram_count

# Define trigrams to calculate
trigrams_to_calculate = [
    ('in', 'the', 'past'),
    ('in', 'the', 'time'),
    ('the', 'jury', 'said'),
    ('the', 'jury', 'recommended'),
    ('jury', 'said', 'that'),
    ('agriculture', 'teacher', ',')
]

# Calculate and print probabilities
probabilities = []
for w1, w2, w3 in trigrams_to_calculate:
    unsmoothed_prob = trigram_probability(w1, w2, w3, bigram_freq, trigram_freq, vocab_size, alpha = 0.1, smoothed=False)
    smoothed_prob = trigram_probability(w1, w2, w3, bigram_freq, trigram_freq, vocab_size, alpha = 0.1, smoothed=True)
    probabilities.append({
        'trigram': f'{w1} {w2} {w3}',
        'unsmoothed': unsmoothed_prob,
        'smoothed': smoothed_prob
    })

print(probabilities)





