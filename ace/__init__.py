# encoding: utf-8
"""
Base classes of Ace
"""

import os
import glob
import time
import math
import codecs
import itertools

from sklearn.naive_bayes import MultinomialNB
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
        self.vectorizer = HashingVectorizer(
            decode_error='ignore',
            n_features=2**18,
            non_negative=True,
            norm=None,
            analyzer=text_to_vector
        )
        self.classifier = SGDClassifier(loss='log')
        self.labelizer = None

        self.training_path = os.path.expanduser('{}/train'.format(data_path))
        self.training_documents = glob.glob('{}/*.txt'.format(self.training_path))
        self.training_keywords = glob.glob('{}/*.key'.format(self.training_path))

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

        number_of_batches = int(math.ceil(len(document_list) / batch_size))

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

    def train(self, X_training_data=None, y_training_data=None, batch_size=1):
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

        self.classifiers = {i: SGDClassifier() for i in classes}

        for documents, keywords in itertools.izip_longest(
                self.document_batch(
                    document_list=X_training_data,
                    batch_size=batch_size),
                self.document_batch(
                    document_list=y_training_data,
                    batch_size=batch_size)
        ):
            t_start = time.time()

            X_train = self.vectorizer.transform(documents)

            # [
            #  [1, 30, 40, 1, 3, 50],
            #  [30, 20, 1, 30, 40, 50]
            # ]
            print([[k for k in key.split('\n') if k != ''] for key in keywords])
            y_train = self.labelizer.fit_transform(
                [[k for k in key.split('\n') if k != ''] for key in keywords]
            )
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
                    classes=[label, u'not {}'.format(label)]
                )

            t_stop = time.time()

            self.training['n_train'] += 1
            self.training['total_time_taken'] += (t_stop - t_start)

    def predict(self, X):
        """
        Prediction for X using one-vs-rest, multilabel classification
        :param X: data to put in the function
        """
        labels = []
        for label in self.classifiers:
            # print self.classifiers[label].predict(X)
            true_labels = [i for i in self.classifiers[label].predict(X) if i != 'not {}'.format(label)]
            labels.extend(true_labels)

        return labels

