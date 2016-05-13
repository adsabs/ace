import os
import glob
import codecs
import argparse

from ace import Corpus
from cPickle import Unpickler


def fit_corpus():

    # Fit the corpus on training data
    corpus = Corpus(
        data_path='.',#'r_and_d/data/'
    )

    if os.path.isfile('models/classifiers.m'):

        with open('models/classifiers.m') as f:
            pickle = Unpickler(f)
            corpus.classifiers = pickle.load()
    else:
        print('Training')
        corpus.train(batch_size=50, n_iter=5)
        # corpus.save()

    # Try it on the test data
    test_text, test_keyword = [], []
    for text in glob.glob('r_and_d/data/test/2015MNRAS.454.2173R.txt'):
        with codecs.open(text, 'r', 'utf-8') as f:
            test_text.append(f.read())

        keyword = text.replace('.txt', '.key')
        with codecs.open(keyword, 'r', 'utf-8') as f:
            test_keyword.append([i.strip() for i in f.readlines() if i != ''])

    test_X = corpus.vectorizer.transform(test_text)
    pred_Y = corpus.predict(test_X)

    recall = corpus.recall(pred_Y, test_keyword)
    precision = corpus.precision(pred_Y, test_keyword)

    for exp, pred in zip(test_keyword, pred_Y):
        print('Expected: {}'.format(exp))
        print('Predicted: {}'.format(pred))

    print('Overlap: {}'.format([i for i in pred if i in exp]))

    print('Recall: {}'.format(recall))
    print('Precision: {}'.format(precision))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', action='store_true')
    args = parser.parse_args()

    if args.train:
        fit_corpus()
    else:
        parser.print_help()
