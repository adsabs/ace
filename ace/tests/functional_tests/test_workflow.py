"""
Simple functional tests to test the workflow
"""

import mock

from ace import Corpus
from unittest import TestCase, skip


class TestCorpus(TestCase):
    """
    Test the pipeline for supervised learning
    """

    def setUp(self):
        """
        Setup tests
        """
        self.corpus = Corpus(
            data_path='ace/tests/stub_data'
        )

    def test_list_of_documents(self):
        """
        Test it finds the correct list of training documents
        """

        docs = self.corpus.training_documents
        docs.sort()
        self.assertEqual(
            docs,
            ['ace/tests/stub_data/train/train_01.txt',
             'ace/tests/stub_data/train/train_02.txt',
             'ace/tests/stub_data/train/train_03.txt',
             'ace/tests/stub_data/train/train_04.txt',
             'ace/tests/stub_data/train/train_05.txt',
             'ace/tests/stub_data/train/train_06.txt']
        )

    def test_get_batch(self):
        """
        Test getting batches works correctly
        """
        d1 = self.corpus.get_batch(
            document_list=self.corpus.training_documents,
            batch_start=0,
            batch_size=3
        )
        self.assertEqual(len(d1), 3)

        d2 = self.corpus.get_batch(
            document_list=self.corpus.training_documents,
            batch_start=3,
            batch_size=3
        )
        self.assertEqual(len(d2), 3)
        self.assertNotEqual(d1, d2)

    def test_iterate_batch(self):
        """
        Simple test to ensure we iterate batches properly
        """
        counter = 0

        for d in self.corpus.document_batch(
                document_list=self.corpus.training_documents,
                batch_size=2):

            self.assertEqual(len(d), 2)
            counter += 1

        self.assertEqual(counter, 3)

    def test_collecting_classes(self):
        """
        Test we can collect all the classes
        """
        classes = self.corpus.get_classes(
            document_list=self.corpus.training_keywords,
            batch_size=2
        )
        self.assertEqual(len(classes), 8)
        self.assertEqual(list(set(classes)), classes)

    def test_fitting_of_train_via_batches(self):
        """
        Test that the fitting of partial batches works as expected when applied
        to the test data
        """
        self.corpus.train(
            X_training_data=self.corpus.training_documents,
            y_training_data=self.corpus.training_keywords,
            batch_size=2,
            n_iter=3
        )

        keys = self.corpus.training.keys()
        keys.sort()
        self.assertEqual(
            ['accuracy', 'n_train', 'total_time_taken'],
            keys
        )

        self.assertEqual(
            self.corpus.training['n_train'],
            3
        )

        text_X = ['spectroscopy of gamma-ray bursts spectroscopy',
                  'simulations of galaxies']
        test_X = self.corpus.vectorizer.transform(text_X)

        test_Y = [['gamma-ray burst'], ['galaxy']]

        prediction = self.corpus.predict(test_X)

        self.assertTrue(len(prediction) > 0)

        recall = self.corpus.recall(test_Y, prediction)
        self.assertAlmostEqual(recall, 0.75, delta=0.01)

        precision = self.corpus.precision(test_Y, prediction)
        self.assertAlmostEqual(precision, 0.22, delta=0.01)

        fbeta_1 = self.corpus.fbeta_score(precision, recall, beta=1)
        self.assertAlmostEqual(fbeta_1, 0.35, delta=0.01)

    def test_save_model(self):
        """
        Test that we can save the model to disk
        """
        self.corpus.train(
            X_training_data=self.corpus.training_documents,
            y_training_data=self.corpus.training_keywords,
            batch_size=2
        )

        m = mock.mock_open()
        with mock.patch('ace.open', m, create=True):
            self.corpus.save()

        m.assert_called_once_with('models/classifiers.m', 'w')

    #
    # @mock.patch('ace.manage.codecs.open')
    # def test_workflow(self, mock_codecs):
    #     """
    #     Test workflow
    #     """
    #
    #     # 1. We want to load all documents, and convert their content from text
    #     #    into a sparse matrix of bag-of-words, which is saved to disk in a
    #     #    nice format. If they have keywords, then we also do it for the
    #     #    keywords too.
    #     document_iterator = [
    #         'ace/tests/stub_data/train_01.txt',
    #         'ace/tests/stub_data/train_02.txt',
    #         'ace/tests/stub_data/train_03.txt',
    #         'ace/tests/stub_data/train_04.txt',
    #         'ace/tests/stub_data/train_05.txt',
    #         'ace/tests/stub_data/train_06.txt',
    #     ]
    #     mock_docs = []
    #     for document in document_iterator:
    #         with open(document, 'r') as f:
    #             mock_docs.append(f.read())
    #
    #     file_instance = mock_codecs.return_value
    #     file_instance.__enter__.return_value = file_instance
    #     file_instance.__exit__.return_value = True
    #     file_instance.read.side_effect = mock_docs
    #     file_instance.write.return_value = True
    #
    #     corpus = Corpus()
    #     corpus.fit()
    #
    #     self.assertEqual(file_instance.write.call_count, 2)
    #     self.assertEqual(file_instance.read.call_count, 6)
    #
    #     self.assertIn(
    #         mock.call('models/corpus_hash_vector.pkl', 'w', 'utf-8'),
    #         mock_codecs.call_args_list
    #     )
    #
    #     # 2. We want to load all bag-of-words matrices, and keywords and
    #     #    determine the TF-IDF of them all. The TF-IDF can be saved to disk
    #     #    also, incase we want to re-fit using a different method.
    #     self.assertIn(
    #         mock.call('models/corpus_tfidf_matrix.pkl', 'w', 'utf-8'),
    #         mock_codecs.call_args_list
    #     )
    #
    #
    #     # 3. We fit the data using a partial_fit method, and then store the fit
    #     #    to disk. We need some definition of training and testing set at
    #     #    this stage, which should be easily extensible.
    #
    #     # 4. We will want to test the fit and store the results to disk also.
    #
    #     # 5. The webapp will load the model, if the document has not been seen
    #     #    before, it will have to create the spare matrix. It will then
    #     #    create a prediction based on the input and the model.
