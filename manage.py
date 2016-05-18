# encoding: utf-8
"""
Manage scripts for running Ace
"""
import sys
import time
import numpy
import logging
import argparse
import matplotlib.pyplot as plt

from ace import Corpus
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Global logger for this module
logger = logging.getLogger(u'Ace')


def fit_corpus(data_path, save=False, load=False, model_path=u'models/classifiers.m'):

    # Get the logger

    # Fit the corpus on training data
    corpus = Corpus(
        data_path=data_path
    )

    if load:
        try:
            corpus.load(model_path)
            logger.info(u'Loaded model: {}'.format(model_path))
        except IOError:
            pass

    if corpus.classifiers is None:
        logger.info(u'Training on batches')
        corpus.train(batch_size=200, n_iter=1)

    if save:
        corpus.save()
        logger.info(u'Saved classifiers to pickle: {}'.format(model_path))

    # Try it on the test data
    test_x = corpus.vectorizer.transform(corpus.test_X_raw)
    pred_y = corpus.predict(test_x)

    logger.info(u'Classes: {}'.format(', '.join(corpus.classifiers.keys())))
    for label in corpus.classifiers:

        logger.info(u'Label: {}'.format(label))
        logger.info(u'----------')

        predict_y = corpus.predictions[label]
        truth_y = [1 if label in i else 0 for i in corpus.test_Y_raw]

        logger.info(u'Confusion matrix:\n {}'
                    .format(confusion_matrix(truth_y, predict_y, labels=[0, 1])))
        logger.info(u'Classification report:\n {}'
                    .format(classification_report(truth_y, predict_y, labels=[0, 1])))
        logger.info(u'Accuracy: {}'
                    .format(accuracy_score(truth_y, predict_y)))
        logger.info(u'===============')
        logger.info(u'')

    recall = corpus.recall(pred_y, corpus.test_Y_raw)
    precision = corpus.precision(pred_y, corpus.test_Y_raw)

    logger.info(u'Training details: {}'.format(corpus.training))
    logger.info(u'Average recall: {}'.format(recall))
    logger.info(u'Average precision: {}'.format(precision))


def investigate_train_size(data_path, save=False, load=False):
    """
    Make a plot of the recall and precision against the training size of the
    data
    :param data_path: path to the data
    :param save: should the method save the classifier to disk
    :param load: should the method load the classifier from disk
    """
    corpus = Corpus(
        data_path=data_path
    )
    test_x = corpus.vectorizer.transform(corpus.test_X_raw)

    training_sizes = numpy.arange(0.05, 0.95, 0.05)

    precisions = []
    recalls = []
    times = []

    for train_size in training_sizes:
        logger.info(u'Training with size: {} [%]'.format(train_size))

        t1 = time.time()

        model_path = u'models/classifiers_size{}pc.m'.format(train_size)
        if load:
            try:
                corpus.load(model_path=model_path)
                logger.info(u'Loaded model: {}'.format(model_path))
            except IOError:
                corpus.train(
                    batch_size=200,
                    n_iter=1,
                    training_size=train_size
                )
        else:
            corpus.train(
                    batch_size=200,
                    n_iter=1,
                    train_size=train_size
                )

        t2 = time.time()
        times.append(
            t2 - t1
        )

        pred_y = corpus.predict(test_x)

        precisions.append(
            corpus.precision(pred_y, corpus.test_Y_raw)
        )
        recalls.append(
            corpus.recall(pred_y, corpus.test_Y_raw)
        )

        if save:
            corpus.save(mode_path=model_path)


    # Make the plot
    fig = plt.figure(0)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)

    ax1.errorbar(training_sizes, recalls)
    ax1.set_ylabel(u'Recall')

    ax2.errorbar(training_sizes, precisions)
    ax2.set_ylabel(u'Precision')

    ax3.errorbar(training_sizes, times)
    ax3.set_ylabel(u'Time taken [s]')
    ax3.set_xlabel(u'Training size [%]')

    plt.savefig(u'images/training_size.png', format=u'png')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', action='store_true', default=False)
    parser.add_argument('-i', '--investigate-train-size', dest='train_size', action='store_true', default=False)
    parser.add_argument('-d', '--data', dest='data', default=False)
    parser.add_argument('--save', dest='save', action='store_true', default=False)
    parser.add_argument('--load', dest='load', action='store_true', default=False)
    args = parser.parse_args()

    if not args.data:
        parser.print_help()
        sys.exit()

    if args.train:
        fit_corpus(data_path=args.data, save=args.save, load=args.load)
    elif args.train_size:
        investigate_train_size(data_path=args.data, save=args.save, load=args.load)
    else:
        parser.print_help()
