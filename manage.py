import codecs
import argparse

from ace import Corpus


def fit_corpus():

    corpus = Corpus(
        data_path='r_and_d/data'
    )

    corpus.train(batch_size=100)

    with codecs.open('r_and_d/data/test/2015MNRAS.454.2003P.txt', 'r', 'utf-8') as f:
        test_text = f.read()

    with codecs.open('r_and_d/data/test/2015MNRAS.454.2003P.key', 'r', 'utf-8') as f:
        test_keyword = [i.strip() for i in f.readlines() if i != '']

    print(test_keyword)

    test_X = corpus.vectorizer.transform(test_text)
    print(corpus.training)
    print('Predicted: {}'.format(corpus.predict(test_X)))
    print('Actual: {}'.format(test_keyword))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', action='store_true')
    args = parser.parse_args()

    if args.train:
        fit_corpus()
    else:
        parser.print_help()
