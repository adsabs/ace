# encoding: utf-8
"""
Building a prototype for text classification
"""

import glob
import time
import codecs

from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer


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


if __name__ == '__main__':

    t_start = time.time()

    # Load the training data
    ft_files = glob.glob('data/*.txt')[0:100]
    text = []
    keywords = []
    for ft in ft_files:
        with codecs.open(ft, 'r', 'utf-8') as f:
            text.append(f.read())

        with codecs.open(ft.replace('.txt', '.key'), 'r', 'utf-8') as f:
            keywords.append(f.read().split('\n'))

    # Lets use 1 of the documents as a test
    text_test = text.pop(-1)
    keywords_test = keywords.pop(-1)

    # We want to convert the labels into vectors. For example, if we have:
    # keywords = [
    #             ['solar', 'physics', 'astronomy'],
    #             ['physics', 'lasers'],
    #             ['astronomy']
    #           ]
    # this would become:
    # keywords_binarised = [
    #             [1, 1, 1, 0],
    #             [0, 1, 0, 1],
    #             [0, 0, 1, 0]
    #           ]
    mlb = MultiLabelBinarizer()
    mlb.fit(keywords)
    keywords_vector = mlb.transform(keywords)

    # We generate a transform from words -> vector space. This is very similar
    # to the above conversion of the keywords. In this scenario, the entire
    # corpus from our training set is converted into an id -> word sparse-
    # matrix.
    bow_transform = CountVectorizer(analyzer=text_to_vector).fit(' '.join(text))

    # We transform our corpus into the unique vector space
    bow_vector = bow_transform.transform(text)

    # We convert the vector into a term frequency - inverse document frequency
    # Term frequencey: f_t (number of times in a document term t exists)
    # Inverse document frequency: log(N/n_t) (number of documents divided by
    #                                         the number of documents that
    #                                         contain term t)
    # TF-IDF: f_t * log(N/n_t)
    #
    # This selects words that are not common throughout the corpus, but have a
    # high frequency in a document. This is a heuristic measure.
    #
    tfidf_transformer = TfidfTransformer().fit(bow_vector)
    tfidf_bow = tfidf_transformer.transform(bow_vector)

    # We then fit using some technique, eg., Naive Bayes or SVM
    # We want to see whether a given keyword is probable, based on the input
    # bag-of-words vector space given.
    print 'TF-IDF Bag-of-Words', tfidf_bow.shape
    print 'Keywords Binary Matrix', keywords_vector.shape
    one_vs_rest_classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    one_vs_rest_classifier.fit(X=tfidf_bow, y=keywords_vector)

    # Now do a prediction to see what our testing data looks like
    text_test = [text_test]
    bow_test = bow_transform.transform(text_test)
    vector_test = tfidf_transformer.transform(bow_test)
    prediction = one_vs_rest_classifier.predict(X=vector_test)

    t_finish = time.time()

    for item, labels in zip(text_test, prediction):
        print 'Text:'
        print item, '\n'
        print 'Predicted labels:', mlb.classes_[labels == 1]

    print 'Human labels:', keywords_test

    print 'Keyword array shape: ', len(mlb.classes_)
    print 'Sparse training matrix shape: ', bow_vector.shape
    print '-----------------------------------\n'
    print 'Time taken: {} seconds'.format(t_finish - t_start)

