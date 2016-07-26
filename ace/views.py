# encoding: utf-8
"""
Views
"""
import json
import codecs

from utils import get_post_data
from flask import request, current_app, Response
from flask.ext.restful import Resource
from jinja2 import Template


class ConceptView(Resource):
    """
    End point to receive a list of keywords or concepts, based on a document
    or list of documents received.
    """

    def post(self):
        """
        HTTP POST request that returns a list of concepts or keywords, based on
        the user given string

        Post data
        ---------
        abstract: document's abstract

        Return data (on success)
        ------------------------
        concepts: <list>

        HTTP Responses:
        --------------
        Succeed authentication: 200
        Any other responses will be default Flask errors
        """
        post_data = get_post_data(request, types={'abstract': unicode})
        current_app.logger.debug('Received data from user: {}'.format(post_data))

        corpus = current_app.config['CORPUS']
        x_predict = corpus.vectorizer.transform([post_data['abstract']])
        labels = [list(i) for i in corpus.predict(x_predict)]

        current_app.logger.debug('Found labels: {}'.format(labels))

        return {'concepts': labels}, 200


class ReportView(Resource):
    """
    Serve a static page with report information, only intended for use by
    adminstrators
    """

    def get(self):
        """
        Return static page of labels and their fits
        """
        with codecs.open('models/meta_data.json', 'r', 'utf-8') as f:
            meta_data = json.load(f)

        with codecs.open('ace/templates/report.html', 'r', 'utf-8') as f:
            template = Template(f.read())

        return Response(
            template.render(meta_data=meta_data),
            mimetype='text/html'
        )

