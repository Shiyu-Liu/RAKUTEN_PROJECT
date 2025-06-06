import sys
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

# Sample text

def visualize_wordcloud(df):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    filtered_words = []
    # Preprocess text
    for text in df['text']:
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        words = [word for word in words if word not in stop_words]
        filtered_words.extend(words)

    # count word frequencies
    word_freq = Counter(filtered_words)

    # WORD CLOUD
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # BAR CHART
    top_words = word_freq.most_common(10)  # Top 10 words
    words, counts = zip(*top_words)

    total_count = sum(word_freq.values())
    percentages = [count / total_count * 100 for count in counts]

    _, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.bar(words, counts, color=color)
    ax1.set_ylabel('Frequency', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("Top 10 Most Frequent Words with Percentages")
    ax1.set_xticklabels(words, rotation=45)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(words, percentages, color=color, marker='o', linestyle='--')
    ax2.set_ylabel('Percentage (%)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Please provide the csv file of the preprocessed data")
        return 0
    file = sys.argv[1]

    df = pd.read_csv(file, delimiter=";")
    visualize_wordcloud(df)

if __name__=="__main__":
    main()