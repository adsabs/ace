"""
Test manage scripts
"""

import mock

from unittest import TestCase

from ace.manage import doc_to_sparse_matrix


class TestManage(TestCase):
    """
    Test the pipeline for supervised learning
    """

    def test_docs_to_sparse_matrix(self):
        """
        Test the conversion of a training corpus into a HashVector, and also
        store each document's sparse matrix to disk (?)
        """


