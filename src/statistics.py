import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from src.util import *
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from src.processing import preprocess
import os

graph_path = "../graphs/"
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    :param rects: bars
    :param ax: axes
    :return: nothing (inplace function)
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.03 * height,
                '%d' % int(height),
                ha='center', va='bottom', fontweight='bold')

def plot_class_distribution(tweets, labels):
    """
    Plot distribution of tweets across classes
    :param tweets: tweets list (strings)
    :param labels: labels list (strings)
    :return: count by class
    """
    check_data_format(tweets, labels)
    # if labels[0].isdigit():
    #     labels.sort(key=int)
    # labels_count = Counter(labels)
    # else:
    labels_count = Counter(sorted(labels))
    # labels_count = OrderedDict(sorted(labels_count.items()))
    classes = labels_count.keys()
    bar_x_locations = np.arange(len(classes))

    w=0.25
    plt.bar(bar_x_locations, labels_count.values(), width=w, align = 'center', label = 'tweets')
    plt.xticks(bar_x_locations, classes)
    plt.legend()
    plt.savefig(graph_path + 'class_distribution.png')
    plt.show()
    return labels_count

def plot_best_features(tweets, labels, columns=None, n=15, score_func=chi2, figsize = (15, 15)):
    """
    Show a bar plot displaying best features for classification according to the score function provided
    :param tweets: list of tweets (strings) or 2D array IF columns provided
    :param labels: list of labels (strings)
    :param columns: names of each features, in the right order. If not, provided, a BOW is extracted from assumed tweets
    :param n: number of best features to plot
    :param score_func: score function to use (chi2, mutual_info...)
    :param figsize: figure size
    :return: DataFrame corresponding to sum of the n features values across tweets for each label (category)
    """
    check_data_format(tweets, labels)
    labels = [str(l) for l in labels]
    if not columns:
        vectorizer = CountVectorizer(binary = False, ngram_range=(1, 1), tokenizer=lambda text: preprocess(text, handle_negation=False,
                                                                                           lowercase=True))
        X = vectorizer.fit_transform(tweets, labels)
        columns = vectorizer.get_feature_names()
    else:
        X = tweets
    try:
        selector = SelectPercentile(score_func=chi2, percentile=100)
        X_reduced = selector.fit_transform(X, labels)
        best_features_names_ordered = np.array(columns)[np.argsort(selector.scores_)][::-1]
        dataframe = convert_to_dataframe(X, labels, columns=columns)
        indices = np.arange(n)
        counts = (dataframe.groupby(['label'])[best_features_names_ordered[i]].sum() for i in indices)
        counts = pd.DataFrame(counts)
        try:
            int(labels[0])
            labels = list(sorted(set(labels), key=int))
        except:
            labels = list(sorted(set(labels)))
        fig, ax = plt.subplots(figsize=figsize)
        width = 0.25
        ax.set_ylabel('Scores')
        ax.set_title('Scores by word and sentiment')
        ax.set_xticks(indices)
        ax.set_xticklabels((x for x in counts.index))
        if len(labels) == 3:
            s1 = plt.bar(indices - width, counts[labels[0]], width, color='red')
            s2 = plt.bar(indices, counts[labels[1]], width, color='blue')
            s3 = plt.bar(indices + width, counts[labels[2]], width, color='green')
            plt.legend((s1[0], s2[0], s3[0]), (labels[0], labels[1], labels[2]))
            autolabel(s1, ax)
            autolabel(s2, ax)
            autolabel(s3, ax)
        elif len(labels) == 2:
            s1 = plt.bar(indices - width/2, counts[labels[0]], width, color='red')
            s2 = plt.bar(indices + width/2, counts[labels[1]], width, color='green')
            plt.legend((s1[0], s2[0]), (labels[0], labels[1]))
            autolabel(s1, ax)
            autolabel(s2, ax)
        elif len(labels) == 5:
            width = 0.15
            s1 = plt.bar(indices - width*2, counts[labels[0]], width, color='red')
            s2 = plt.bar(indices-width, counts[labels[1]], width, color='orange')
            s3 = plt.bar(indices , counts[labels[2]], width, color='b')
            s4 = plt.bar(indices + width, counts[labels[3]], width, color='lightgreen')
            s5 = plt.bar(indices + width*2, counts[labels[4]], width, color='green')
            plt.legend((s1[0], s2[0], s3[0], s4[0], s5[0]), (labels[0], labels[1], labels[2], labels[3], labels[4]))
            autolabel(s1, ax)
            autolabel(s2, ax)
            autolabel(s3, ax)
            autolabel(s4, ax)
            autolabel(s5, ax)

        fig.autofmt_xdate()


        plt.savefig(graph_path + 'best_features.png')
        plt.show()
        return counts
    except ValueError as e:
        print(e)
        print('Could not plot, incorrect data type transmitted')

def plot_word_distribution(tweets, labels, word):
    """
    Plot the distribution of the given word over the classes (raw term count). Works only with unigram words.
    :param tweets: list of tweets (strings)
    :param labels: list of labels (strings)
    :param word: word to plot
    :return: counts of the word
    """
    check_data_format(tweets, labels)
    vectorizer = CountVectorizer(ngram_range=(1, 1), tokenizer=lambda text: preprocess(text, handle_negation=False,
                                                                                       lowercase=True))
    X = vectorizer.fit_transform(tweets)
    if word not in vectorizer.get_feature_names():
        return "The word {0} is not in the vocabulary".format(word)
    word_index = vectorizer.vocabulary_[word]
    dataframe = convert_to_dataframe(X[:,word_index], labels, columns=[word])
    # remove irrelevant topic that may be found in the dataset

    if 'off topic' in set(labels):
        dataframe = dataframe[dataframe.label != 'off topic']

    counts = dataframe.groupby('label').agg({word: sum})
    classes = counts.index
    bar_x_locations = np.arange(len(classes))
    w=0.25
    plt.bar(bar_x_locations, counts.values.ravel(), width=w, align = 'center', label = word)
    plt.xticks(bar_x_locations, classes)
    plt.legend()

    plt.savefig(graph_path + 'word_distribution.png')
    plt.show()

    return counts