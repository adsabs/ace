# encoding: utf-8
"""
Views
"""

from utils import get_post_data
from flask import request
from flask.ext.restful import Resource


class ConceptView(Resource):
    """
    End point to receive a list of keywords or concepts, based on a document
    or list of documents received.
    """

    def post(self):
        """
        HTTP POST request that returns a list of concepts or keywords, based on
        the user given document(s)

        Post data
        ---------
        bibcode: <list>

        Return data (on success)
        ------------------------
        concepts: <list>

        HTTP Responses:
        --------------
        Succeed authentication: 200
        Any other responses will be default Flask errors
        """

        post_data = get_post_data(request, types={'bibcode': list})

        return {'concepts': []}, 200
