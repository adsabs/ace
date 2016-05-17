# encoding: utf-8
"""
Manage scripts for running Ace
"""
import os
import sys
import glob
import codecs
import argparse

from ace import Corpus
from cPickle import Unpickler
from sklearn.metrics import confusion_matrix, accuracy_score


def fit_corpus(data_path):

    # Fit the corpus on training data
    corpus = Corpus(
        data_path=data_path
    )

    if os.path.isfile('models/classifiers.m'):

        with open('models/classifiers.m') as f:
            pickle = Unpickler(f)
            corpus.classifiers = pickle.load()
    else:
        print('Training')
        corpus.train(batch_size=50, n_iter=10)

    # Try it on the test data
    test_text, test_keyword = [], []
    for text in glob.glob('data/GRBs/test/*.txt'):
        with codecs.open(text, 'r', 'utf-8') as f:
            test_text.append(f.read())

        keyword = text.replace('.txt', '.key')
        with codecs.open(keyword, 'r', 'utf-8') as f:
            test_keyword.append([i.strip() for i in f.readlines() if i != ''])

    test_X = corpus.vectorizer.transform(test_text)
    pred_Y = corpus.predict(test_X)

    cm = confusion_matrix(test_keyword, pred_Y)
    recall = corpus.recall(pred_Y, test_keyword)
    precision = corpus.precision(pred_Y, test_keyword)

    print('Confusion matrix:\n{}'.format(cm))
    print('Accuracy: {}'.format(accuracy_score(test_keyword, pred_Y)))
    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', action='store_true')
    parser.add_argument('-d', '--data', dest='data', default=False)
    args = parser.parse_args()

    if not args.data:
        parser.print_help()
        sys.exit()

    if args.train:
        fit_corpus(data_path=args.data)
    else:
        parser.print_help()
