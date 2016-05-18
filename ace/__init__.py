# encoding: utf-8
"""
Base classes of Ace
"""

import os
import glob
import time
import math
import numpy
import codecs
import itertools

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor

from textblob import TextBlob
from nltk.corpus import stopwords


def text_to_vector(fulltext):
    """
    Small method that removes stop-words, lowercases and converts words to
    their base lemmas.
    """
    # Load English stop-words
    stop_words = set(stopwords.words('english'))

    # Take the base word (lemma) of each word
    bag_of_words = [i.lemma for i in TextBlob(fulltext).words]

    # Parts-of-speech: get all the words in lower case
    bag_of_words = [i.lower() for i in bag_of_words]

    # Remove all of the stop words
    bag_of_words = [word for word in bag_of_words if word not in stop_words]

    return bag_of_words


class Corpus(object):
    """
    Class to carry out out-of-core fitting of text documents, to find their
    relevant keywords.
    """

    def __init__(self, data_path='ace/data'):

        # Define the hash vector and labelizer
        self.vectorizer = HashingVectorizer(
            encoding='utf-8',
            binary=False,
            non_negative=True,
            n_features=2**18,
            analyzer=text_to_vector
        )
        self.labelizer = None

        # Find the training data
        self.training_path = os.path.expanduser('{}/train'.format(data_path))
        self.testing_path = os.path.expanduser('{}/test'.format(data_path))

        docs = glob.glob('{}/*.txt'.format(self.training_path))
        self.training_documents = []
        self.training_keywords = []
        for i in range(len(docs)):
            self.training_documents.append(docs[i])
            self.training_keywords.append(docs[i].replace('.txt', '.key'))

        # Find the test data
        docs = glob.glob('{}/*.txt'.format(self.testing_path))
        self.testing_documents = []
        self.testing_keywords = []
        for i in range(len(docs)):
            self.testing_documents.append(docs[i])
            self.testing_keywords.append(docs[i].replace('.txt', '.key'))

        # Load the contents of the raw test data
        self.test_X_raw = []
        self.test_Y_raw = []
        for text in self.testing_documents:
            with codecs.open(text, 'r', 'utf-8') as f:
                self.test_X_raw.append(f.read())

            keyword = text.replace('.txt', '.key')
            with codecs.open(keyword, 'r', 'utf-8') as f:
                self.test_Y_raw.append([i.strip() for i in f.readlines() if i != ''])

        self.training = dict()

    def get_batch(self, document_list=[], batch_start=0, batch_size=1):
        """
        Get a batch of documents and their keywords
        :param document_list: documents to load batches from
        :type document_list: list
        :param batch_start: where to start the batch
        :type batch_start: int
        :param batch_size: size of the batch
        :type batch_size: int

        :return: tuple(list, list)
        """
        documents = []
        for document in document_list[batch_start: batch_start+batch_size]:
            with codecs.open(document, 'r', 'utf-8') as f:
                documents.append(f.read())
        return documents

    def document_batch(self, document_list=[], batch_size=1):
        """
        Iterator for going through batches of documents
        :param document_list: documents to load batches from
        :type document_list: list
        :param batch_size: size of batch
        :type batch_size: int
        :return: iterator
        """

        number_of_batches = int(math.ceil(len(document_list) / (batch_size*1.0)))

        for i in range(number_of_batches):
            yield self.get_batch(
                document_list=document_list,
                batch_size=batch_size
            )

    def get_classes(self, document_list=[], batch_size=1):
        """
        Get all the classes that y_train can be
        :param document_list: documents to load batches from
        :type document_list: list
        :param batch_size: size of batch
        :type batch_size: int

        :return: sparse-matrix
        """
        hashmap = {}
        for keywords in self.document_batch(
                document_list=document_list,
                batch_size=batch_size):
            [[hashmap.setdefault(k, '') for k in keys.split('\n') if k != ''] for keys in keywords]

        return hashmap.keys()

    def train(self, X_training_data=None, y_training_data=None, batch_size=1, n_iter=1):
        """
        Use the training documents to fit a classifier.
        """
        if X_training_data is None:
            X_training_data = self.training_documents

        if y_training_data is None:
            y_training_data = self.training_keywords

        classes = self.get_classes(
            document_list=self.training_keywords,
            batch_size=batch_size
        )

        self.labelizer = MultiLabelBinarizer(classes=classes)

        self.training['n_train'] = 0
        self.training['total_time_taken'] = 0
        self.training['accuracy'] = []

        self.classifiers = {i: SGDClassifier(loss='log') for i in classes}

        t_start = time.time()
        for i in range(n_iter):
            for documents, keywords in itertools.izip_longest(
                    self.document_batch(
                        document_list=X_training_data,
                        batch_size=batch_size),
                    self.document_batch(
                        document_list=y_training_data,
                        batch_size=batch_size)
            ):

                X_train = self.vectorizer.transform(documents)

                # [
                #  [1, 30, 40, 1, 3, 50],
                #  [30, 20, 1, 30, 40, 50]
                # ]
                y_train = self.labelizer.fit_transform(
                    [[k for k in key.split('\n') if k != ''] for key in keywords]
                )
                # print y_train
                # i (row): single document
                # j (col): keyword
                # [
                #   [1, 0, 1, 0],
                #   [0, 1, 1, 1]
                # ]
                # Want to pass a block X of documents, with a single keyword, 1 or 0
                # For example, the first classifier would receive:
                # X = [
                #  [1, 30, 40, 1, 3, 50]
                #  [30, 20, 1, 30, 40, 50]
                # ]
                # Y = [1, 0, 1, 0]

                for label in classes:
                    y_t = y_train[:, self.labelizer.classes.index(label)]
                    self.classifiers[label].partial_fit(
                        X=X_train,
                        y=y_t,
                        classes=[0, 1]
                    )

            self.training['n_train'] += 1

        t_stop = time.time()
        self.training['total_time_taken'] += (t_stop - t_start)

    def predict(self, X):
        """
        Prediction for X using one-vs-rest, multilabel classification
        :param X: data to put in the function
        """

        self.predictions = {}

        for label in self.classifiers:
            self.predictions[label] = self.classifiers[label].predict(X)

        labels = []
        for i in range(X.shape[0]):
            labels.append([
                l for l in self.classifiers if self.predictions[l][i] == 1
            ])

        # labels = []
        #
        # for x in X:
        #     x_labels = []
        #
        #     for label in self.classifiers:
        #
        #         p = self.classifiers[label].predict(x)
        #         if p == 1:
        #             x_labels.append(label)
        #
        #     labels.append(numpy.array(x_labels))

        return numpy.array(labels)

    def recall(self, Y_exp, Y_pred):
        """
        Recall
        :param Y_exp:
        :param Y_pred:
        :return:
        """
        true_positive, false_negative, recall = [], [], []
        for ye, yp in zip(Y_exp, Y_pred):
            true_positive.extend([1 for i in yp if i in ye])
            false_negative.extend([1 for i in ye if i not in yp])

            tp = sum(true_positive)*1.0
            fn = sum(false_negative)*1.0

            recall.append(
                tp / (tp + fn)
            )

        return sum(recall) / len(recall)*1.0

    def precision(self, Y_exp, Y_pred):
        """
        Precision
        :param Y_exp:
        :param Y_pred:
        :return:
        """
        true_positive, false_positive, precision = [], [], []

        for ye, yp in zip(Y_exp, Y_pred):
            true_positive.extend([1 for i in yp if i in ye])
            false_positive.extend([1 for i in yp if i not in ye])

            tp = sum(true_positive)*1.0
            fp = sum(false_positive)*1.0

            precision.append(
                tp / (tp + fp)
            )

        return sum(precision) / len(precision)*1.0

    def fbeta_score(self, precision, recall, beta):
        """
        F-beta score
        :param precision:
        :param recall:
        :param beta:
        :return:
        """
        return (1+beta**2)*(precision*recall)/(beta**2*precision + recall)

    def save(self):
        """
        Save model to disk
        """
        from cPickle import Pickler

        if self.classifiers:
            with open('models/classifiers.m', 'w') as f:
                pickle = Pickler(f, -1)
                pickle.dump(self.classifiers)

    #
    # def evaluate(self, Y_exp, Y_pred):
    #
    #     precision, recall = [], []
    #     for i, j in zip(Y_exp, Y_pred):
    #
    #         print 'yexp', Y_exp
    #         print 'ypred', Y_pred
    #
    #         precision.append(metrics.precision_score(i, j))
    #         recall.append(metrics.recall_score(i, j))
    #
    #     return precision, recall
