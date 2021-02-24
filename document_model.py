import logging
from spacy_tagger import Tagger

LOGGER = logging.getLogger(__name__)


class Document:
    """
    A document is a combination of text and the positions of the tags in that text.
    """
    tagger = Tagger()

    def __init__(self, text, tags, filename=None):
        """
        :param text: document text as a string
        :param tags: list of Tag objects
        """
        self.tokens = [token.text for token in Document.tagger.tokenize_text(text)]
        self.tags = tags
        self.text = text
        self.filename = filename
