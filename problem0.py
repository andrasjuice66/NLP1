

import nltk
from collections import Counter
import nltk.corpus
import matplotlib.pyplot as plt
brown = nltk.corpus.brown

nltk.download('averaged_perceptron_tagger')
sentences = brown.sents()
word_freq_all = Counter(brown.words())
pos_tags = nltk.pos_tag(brown.words())
num_tokens = len(brown.words())
num_types = len(set(brown.words()))
avg_words_per_sentence = sum(len(sentence) for sentence in sentences) / len(sentences)
avg_word_length = sum(len(word) for word in brown.words()) / num_tokens
pos_counts = Counter(tag for _, tag in pos_tags)
most_common_pos = pos_counts.most_common(10)


sorted_words_all = sorted(word_freq_all.items(), key=lambda x: x[1], reverse=True)



genres = ['news', 'romance']
genre_word_freq = {genre: Counter(brown.words(categories=genre)) for genre in genres}


sorted_words_genres = {
    genre: sorted(freq.items(), key=lambda x: x[1], reverse=True)
    for genre, freq in genre_word_freq.items()
}

(sorted_words_all[:10], {genre: sorted_words[:10] for genre, sorted_words in sorted_words_genres.items()})
print("Number of tokens:", num_tokens)
print("Number of types:", num_types)
print("Number of words:", num_tokens)
print("Average number of words per sentence:", avg_words_per_sentence)
print("Average word length:", avg_word_length)
print("Ten most frequent POS tags:", most_common_pos)


def plot_frequency_curve(data, title, log_scale=False):
    freqs = [freq for _, freq in data]
    labels = range(1, len(freqs) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(labels, freqs, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.savefig(""+title+"log"+str(log_scale)+".png")
    plt.show()
    



genres = ['news', 'romance']
genre_word_freqs = {genre: Counter(brown.words(categories=genre)) for genre in genres}

sorted_freq_all = sorted(word_freq_all.items(), key=lambda x: x[1], reverse=True)
sorted_freq_genres = {genre: sorted(freqs.items(), key=lambda x: x[1], reverse=True) for genre, freqs in genre_word_freqs.items()}


plot_frequency_curve(sorted_freq_all, "Frequency Curve for the Entire Brown Corpus")
plot_frequency_curve(sorted_freq_all, "Log-Log Frequency Curve for the Entire Brown Corpus", log_scale=True)

for genre, freqs in sorted_freq_genres.items():
    plot_frequency_curve(freqs, f"Frequency Curve for {genre.capitalize()} Genre")
    plot_frequency_curve(freqs, f"Log-Log Frequency Curve for {genre.capitalize()} Genre", log_scale=True)