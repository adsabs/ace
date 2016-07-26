# encoding: utf-8
"""
Application factory
"""

import logging.config

from ace import Corpus
from flask import Flask
from flask.ext.watchman import Watchman
from flask.ext.restful import Api
from views import ConceptView, ReportView


def create_app():
    """
    Create the application and return it to the user
    :return: application
    """

    app = Flask(__name__, static_folder='static')
    app.url_map.strict_slashes = False

    # Load config and logging
    load_config(app)
    logging.config.dictConfig(
        app.config['ACE_LOGGING']
    )

    # Load corpus into the application
    corpus = Corpus()
    corpus.load(
        app.config['ACE_MODEL_PATH']
    )
    app.config['CORPUS'] = corpus

    # Register extensions
    Watchman(app, version=dict(scopes=['']))
    api = Api(app)

    # Add the end resource end points
    api.add_resource(ConceptView, '/concept', methods=['POST'])
    api.add_resource(ReportView, '/report', methods=['GET'])
    return app


def load_config(app):
    """
    Loads configuration in the following order:
        1. config.py
        2. local_config.py (ignore failures)
        3. consul (ignore failures)
    :param app: flask.Flask application instance
    :return: None
    """

    app.config.from_pyfile('config.py')

    try:
        app.config.from_pyfile('local_config.py')
    except IOError:
        app.logger.warning('Could not load local_config.py')


if __name__ == '__main__':
    running_app = create_app()
    running_app.run(debug=True, use_reloader=False)
