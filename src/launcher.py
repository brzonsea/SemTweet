#!/usr/bin/env python

"""
This is the main script for scoring, predicting, plotting.
"""

import sys
sys.path.append('../')
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import argparse
from time import time

from src.lexicons import *
from src.embeddings import *

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from src.statistics import *
from sklearn.ensemble import RandomForestClassifier


classifier_choices = ['svm', 'logistic_regression', 'knn', 'naive_bayes_bernoulli', 'naive_bayes_binomial', 'neural_network',
               'tree', 'most_frequent', 'stratified', 'random', 'lexicon', "none"]
classifiers = [SVC(kernel='linear', class_weight='balanced', C=100), LogisticRegression(C=1000, class_weight='balanced'), KNeighborsClassifier(), BernoulliNB(), MultinomialNB(),
               MLPClassifier(), RandomForestClassifier(), DummyClassifier(strategy='most_frequent'),
               DummyClassifier(strategy='stratified'),
               DummyClassifier(strategy='uniform'), ANEWPredictor(), None]
classifiers = dict(zip(classifier_choices, classifiers))

plot_choices = ['chosen word', '$all$', '$classes$']
bag_of_words_choices = ['count', 'binary', 'tfidf']

preprocessor = lambda text: preprocess(text, word_transformation='', lowercase=True)

swn_extractor = SentiWordNet_Extractor()
sentiment140_extractor_uni = Sentiment140ExtractorUnigram()
sentiment140_extractor_bi = Sentiment140ExtractorBigram()
custom_extractor = Custom_Extractor()
anew_extractor = ANEWExtractor()
compiled_lexicon_extractor = CompiledExtractor()
bingliu_extractor = BingLiuExtractor()
afinn_extractor = AFINNExtractor()


def main(arguments):



    begin = time()
    parser = argparse.ArgumentParser(
        description="Train a classifier and either cross validate on training data or score on the test data",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('train', help="Input training data file in format TEXT<tab>LABEL or TEXT<tab>TOPIC<tab>LABEL",
                        type=argparse.FileType('r'))
    parser.add_argument('classifier', choices=classifier_choices,
                        help="Name of the classifier to use on the data: \n{{{}}}".format('|'.join(classifier_choices)),
                        metavar='classifier')
    parser.add_argument('-t', '--test', help="""Input test data file in same format as TRAIN. If not provided, 
                        cross validation is executed on the train data""",
                        type=argparse.FileType('r'), metavar='test_data')
    parser.add_argument('-g', '--graph', help=f"""Plot some distributions across classes, from a specified word, 15 best ones... Test file is ignored. 
{{{'|'.join(plot_choices)}}}""", metavar='plot_type')
    parser.add_argument('-b', '--bag', help="""Whether the bag of words uses counts or binary value. Default binary. \n{{{}}}""".format('|'.join(['count', 'binary'])),
                        metavar = 'bag_of_words_type', choices=bag_of_words_choices)
    parser.add_argument('-p', '--predict',
                        help="""Predict the class of a given sentence, given the training data. If a model has already 
been trained ('model.pkl' by default), then the model is not trained again. So please make sure to delete any pkl 
file that might contain a trained model if you want to retrain. Test data is ignored and no score is output. \n{{{}}}""".format('string to predict'),
                        metavar='string_to_predict')
    parser.add_argument('-w', '--word',
                        help="""Output the most similar words to the one chosen in the corpus after Doc2Vec training.
If 'doc2vec.pkl' file is found in path, model is just loaded""", metavar='word_to_analyze')

    # retrieve arguments
    args = vars(parser.parse_args(arguments))
    train = args['train'].name
    test= args['test'].name if args['test'] else None
    classifier = args['classifier']
    plot_type = args['graph']
    bag_type = args['bag']
    predict_sentence = args['predict']
    doc2vec_word = args['word']
    # load data
    train_tweets, train_labels = load_data(train)
    check_data_format(train_tweets, train_labels)
    print("Number of training tweets : {}".format(len(train_tweets)))

    if plot_type == "$all$":
        plot_best_features(train_tweets, train_labels)
    elif plot_type == "$classes$":
        plot_class_distribution(train_tweets, train_labels)
    elif plot_type:
        result = plot_word_distribution(train_tweets, train_labels, plot_type)
        if isinstance(result, str):
            print(result)
    if bag_type == 'count':
        bag_of_words_extractor = CountVectorizer(binary=False, ngram_range=(1, 3),
                                             tokenizer=lambda text: preprocess(text, word_transformation='',
                                                                               lowercase=True),
                                             lowercase=True)
    elif bag_type == 'tfidf':
        bag_of_words_extractor = TfidfVectorizer(binary=False, ngram_range=(1, 3),
                                             tokenizer=lambda text: preprocess(text, word_transformation='',
                                                                               lowercase=True),
                                             lowercase=True)
    else:
        bag_of_words_extractor = CountVectorizer(binary=True, ngram_range=(1, 3),
                                                 tokenizer=lambda text: preprocess(text, word_transformation='',
                                                                                   lowercase=True),
                                                 lowercase=True)

    binary = CountVectorizer(binary=True, ngram_range=(1, 3),
                                                 tokenizer=lambda text: preprocess(text, word_transformation='',
                                                                                   lowercase=True),
                                                 lowercase=True)
    count = CountVectorizer(binary=False, ngram_range=(1, 3),
                             tokenizer=lambda text: preprocess(text, word_transformation='',
                                                               lowercase=True),
                             lowercase=True)
    tfidf = TfidfVectorizer(binary=False, ngram_range=(1, 3),
                                             tokenizer=lambda text: preprocess(text, word_transformation='',
                                                                               lowercase=True),
                                             lowercase=True)

    # define features list with the chosen bag of words
    extractors = [bag_of_words_extractor, swn_extractor, sentiment140_extractor_uni, sentiment140_extractor_bi, custom_extractor,
                  anew_extractor,
                  compiled_lexicon_extractor, bingliu_extractor, afinn_extractor]
    # enable this line to only use bag of words
    # extractors = [bag_of_words_extractor]

    # this our best feature set. Comment it to enable the bag of words choice in command line.
    extractors = [count, binary, tfidf, swn_extractor, sentiment140_extractor_uni,
                  sentiment140_extractor_bi, custom_extractor,
                  anew_extractor,
                  compiled_lexicon_extractor, bingliu_extractor, afinn_extractor]

    if classifier in ['naive_bayes_bernoulli', 'naive_bayes_binomial']:
        extractors.remove(swn_extractor)
        extractors.remove(afinn_extractor)

    # get chosen classifier
    estimator = classifiers[classifier]
    print("Classifier : {}".format(classifier))

    if doc2vec_word:
        doc2vec = doc2vec_train(train_tweets, train_labels)
        try:
            print("Most similar words to {0} in the train corpus are :\n".format(doc2vec_word))
            for x in doc2vec.most_similar(doc2vec_word):
                    print("{0}, similarity score = {1:.5f}".format(x[0], x[1]))
        except KeyError:
            print("Your word is not in the corpus")
    print("\n")
    if not estimator==None or predict_sentence:
        # create the features union and the pipeline
        features = create_feature_unions(extractors)
        pipeline = create_pipeline(features, estimator)
        if predict_sentence:
            if load_model():
                print("Loading pretrained model...")
                model = load_model()
            elif estimator:
                print("Training model on training data...")
                model = train_model(pipeline, train_tweets, train_labels)
            else:
                print("No model saved and no classifier specified...")
                return
            prediction = predict(model, [predict_sentence])
            print("Your sentence has been predicted as {0}".format(prediction[0]))
            end = time() - begin
            print("Done in {0:.2f}s".format(end))
            return
        # whether fit the model or not (always true excepted is lexicon is the chosen estimator
        fit=True
        # if test data is provided, then train on train and test on test
        if test:
            test_tweets, test_labels = load_data(test)
            check_data_format(test_tweets, test_labels)
            check_compatibility_train_test(train_labels, test_labels)
            print("Number of testing tweets : {}".format(len(test_tweets)))
            if classifier == 'lexicon':
                pipeline = estimator
                print("Lexicon is a particular classifier : it scores directly on the test set\n" +
                      "If used for task C, it will only classify classes -1, 0 and 1")
                fit = False
            training = time()
            predictions, precision, recall, fscore, support, accuracy = evaluate_model_predict(pipeline, train_tweets,
                                                                                               train_labels, test_tweets,
                                                                                               test_labels, fit=fit)
        # if no test data, perform cross validation on train data
        else:
            if classifier == 'lexicon':
                pipeline = estimator
                print("Lexicon is a particular classifier : it scores directly on the whole train set\n" +
                      "If used for task C, it will only classify classes -1, 0 and 1")
                fit = False
            training = time()
            predictions, precision, recall, fscore, support, accuracy = evaluate_model_predict(pipeline, train_tweets,
                                                                                           train_labels, fit=fit)
        end_training = time() - training
        # print results
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('Fscore: {}'.format(fscore))
        print('Support: {}'.format(support))
        # task C
        if all(isinstance(item, int) for item in (set(train_labels))):
            if not test:
                MAE = macro_MAE(train_labels, predictions)
            else:
                MAE = macro_MAE(test_labels, predictions)
            print('Average MAE : {}'.format(MAE))
        # task A or B
        else:
            print('Average fscore : {}'.format((fscore[0] + fscore[-1]) / 2))
            print('Average recall : {}'.format((recall[0] + recall[-1]) / 2))
        print("Accuracy: {mean:.3f}".format(mean=accuracy.mean()))
        print("Training in {0:.2f}s".format(end_training))
    end = time() - begin
    print("Done in {0:.2f}s".format(end))
#


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))






