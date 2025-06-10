import os
import sys
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns

# more filters on stopwords
STOP_WORDS=['de']

def visualize_wordcloud(df):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words.update(STOP_WORDS)

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
    ax1.set_xticklabels(words, rotation=45)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.plot(words, percentages, color=color, marker='o', linestyle='--')
    ax2.set_ylabel('Percentage (%)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(df: pd.DataFrame, df_original: pd.DataFrame):
    labels = df['prdtypecode'].value_counts().sort_index(ascending=True).index
    labels_str = labels.astype(str)

    last_index = df_original.index[-1]
    original_used = df.loc[:last_index, 'prdtypecode'].value_counts().sort_index(ascending=True)
    augmented_counts = df.loc[last_index+1:, 'prdtypecode'].value_counts().sort_index(ascending=True)
    original_unused = df_original['prdtypecode'].value_counts().sort_index(ascending=True) - original_used

    # fill missing values with 0
    original_used = original_used.reindex(labels, fill_value=0)
    augmented_counts = augmented_counts.reindex(labels, fill_value=0)
    original_unused = original_unused.reindex(labels, fill_value=0)

    print(original_unused)
    print(original_used)
    print(augmented_counts)


    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)

    # Plot on both axes
    ax0.bar(labels_str, original_used, color='steelblue', label='Original')
    ax0.bar(labels_str, augmented_counts, bottom=original_used, color='orange', label='Augmented')
    ax0.bar(labels_str, original_unused, bottom=original_used + augmented_counts, color='lightgray', label='Unused')

    ax1.bar(labels_str, original_used, color='steelblue')
    ax1.bar(labels_str, augmented_counts, bottom=original_used, color='orange')
    ax1.bar(labels_str, original_unused, bottom=original_used + augmented_counts, color='lightgray')

    ax0.set_ylim(8000, 11000)
    ax1.set_ylim(0, 5500)

    # Hide x-axis for top plot
    plt.setp(ax0.get_xticklabels(), visible=False)

    ax0.spines['bottom'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax0.tick_params(bottom=False)
    ax1.tick_params(top=False)
    ax1.set_xticklabels(labels_str, rotation=45)
    ax1.set_xlabel("Product Type Code ", fontsize=14)
    ax1.set_ylabel("Number of Samples", fontsize=14)
    ax0.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def visualize_test_set_distribution(df: pd.DataFrame):
    df_counts = df['prdtypecode'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df_counts.index, y=df_counts, palette='viridis')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Product Type Code', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Please provide the path to the data")
        return 0
    dir = sys.argv[1]

    df = pd.read_csv(os.path.join(dir, 'text_data_clean.csv'), delimiter=";", index_col=0)
    visualize_wordcloud(df)

    df_original = pd.read_csv(os.path.join(dir, 'Y_train.csv'), index_col=0)
    visualize_class_distribution(df, df_original)

    df_test_set = pd.read_csv(os.path.join(dir, 'test_set_final_evaluation.csv'), delimiter=";", index_col=0)
    visualize_test_set_distribution(df_test_set)

if __name__=="__main__":
    main()