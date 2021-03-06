# SemTweet

Sentiment Analysis of Twitter posts<br />
Project based on Task 4 from SemEval 2016 : Sentiment Analysis in Twitter.<br />
Subtasks A, B and C have been observed. The program is thus designed for those tasks but is easily adaptable to any text classification task.<br />

## Getting Started

Two file are executable as main :<br />
    launcher.py, the main script for training a model, scoring or predicting a given sentence.<br />
    gridsearch.py has been used for parameter tuning and model selection. The best parameters are by default chosen in launcher.py for each model.<br />

### Prerequisites

Make sure to have scikit-learn, numpy, scipy, nltk, matplotlib, pandas (latest versions) and Python 3+.<br />
File format must be either ```text<tab>label``` (subtask A) or ```text<tab>topic<tab>label```

### Usage

To start the program, for common usage, use launcher.py.
Use either
```
python launcher.py -h
```
or
```
python launcher.py --help
```
to get detailed help on use and argument possible values:

```
launcher.py [-h] [-t test_data] [-g plot_type] [-b bag_of_words_type]
                   [-p string_to_predict] [-w word_to_analyze]
                   train classifier
```

Different functionalities are available. Train data file and classifier name are the two mandatory arguments.<br /><br />
    1. Train and evaluate a model by cross validation on train data, with the classifier specified. Occurs if only train and classifier are specified.<br /><br />
    2. Train the model with specified classifier on train and test on test data, if -t is specified.<br /><br />
    3. Plot different statistics about the train dataset. Can plot class distribution, word distribution, best word features... if -g is specified.<br /><br />
    4. Predict a given string if -p is specified. If a .pkl file containing a trained model exists in path ('model.pkl' by default), the model is used to predict.<br /><br />
    Otherwise, a model is trained on train, and prediction for the sentence is given. In any case, no validation or scoring is done,
    and test data is ignored is specified.<br /><br />
    5. Find the most similar words to the one given in the train data, using doc2vec model, if -w is specified. If 'doc2vec.pkl' is found in path
    then the pre trained model is loaded if possible.<br /><br />
    6. It is possible to combine different arguments (plotting and predicting, plotting and scoring...). Though, when using -g,
    term frenquency-based bag of words is automatically used. Moreover it is impossible to predict a given sentence and to score the model.<br /><br />
    7. "none" can be provided as a classifier, in such a case no model is trained and only if -p is specified a potential pre-trained model
    is loaded and used on the provided sentence. Any scoring process has to retrain the model. Anytime a model is trained (not cross validation), it is saved on disk.<br /><br />
    8. Training may be long depending on the used classifier, because the feature extraction process is heavy. To avoid any bias in the cross validation process,
    the vocabulary creation, the feature extraction and concatenation needed to be included in a heavy pipeline with repetitive processes,
    so that only training data is always used. However, to choose the features one would like to use for classification, it can easily be changed
    in the launcher file, putting the extractors desired :<br /><br />
    ```
    extractors = [bag_of_words_extractor, swn_extractor, sentiment140_extractor_uni, sentiment140_extractor_bi, custom_extractor,
                  anew_extractor,
                  compiled_lexicon_extractor, bingliu_extractor, afinn_extractor]
    ```

Typical examples :<br />
Stratified cross validation scores on train with logistic_regression:
```
python launcher.py ../data/semeval_train_A.txt logistic_regression 
```

Sentiment predicted for "I love NLP", with a count-based bag of words and logistic_regression. 
Train data is either ignored if 'model.pkl' exists or used to train a model:
```
python launcher.py ../data/semeval_train_A.txt logistic_regression -b count -p "I love NLP" 
```
Train logistic regression on train data and test on test:
```
python launcher.py ../data/semeval_train_A.txt logistic_regression ../data/semeval_test_A.txt
```
Plot best 15 discriminative words of train file:
```
python launcher.py ../data/semeval_train_A.txt none -g $all$ 
```
Plot classes distribution of train file:
```
python launcher.py ../data/semeval_train_C.txt none -g $classes$ 
```
Plot distribution of word 'hello' across classes of train file:
```
python launcher.py ../data/semeval_train_A.txt none -g "hello" 
```
Display most similar words to trump in the train file:
```
python launcher.py ../data/semeval_train_A.txt none -w trump
```



## Authors

* **Elliot Bartholme** - https://github.com/elliotbart<br />

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

Saif Mohammad - Sentiment140 Lexicon<br />
University of Florida - ANEW lexicon<br />
beefoo/text-analysis - Compiled lexicon<br />
Informatics and Mathematical Modelling, Technical University of Denmark - AFINN lexicon<br />



