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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
        corpus.train(batch_size=200, n_iter=1)

    # Try it on the test data
    test_x = corpus.vectorizer.transform(corpus.test_X_raw)
    pred_y = corpus.predict(test_x)

    print('Classes: {}'.format(', '.join(corpus.classifiers.keys())))
    for label in corpus.classifiers:

        print('Label: {}'.format(label))
        print('----------')

        predict_y = corpus.predictions[label]
        truth_y = [1 if label in i else 0 for i in corpus.test_Y_raw]

        print('Confusion matrix:\n {}'
              .format(confusion_matrix(truth_y, predict_y, labels=[0, 1])))
        print('Classification report: {}'
              .format(classification_report(truth_y, predict_y, labels=[0, 1])))
        print('Accuracy: {}'
              .format(accuracy_score(truth_y, predict_y)))
        print('\n\n')

    recall = corpus.recall(pred_y, corpus.test_Y_raw)
    precision = corpus.precision(pred_y, corpus.test_Y_raw)

    print('Training details: {}'.format(corpus.training))
    print('Average recall: {}'.format(recall))
    print('Average precision: {}'.format(precision))
#
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
