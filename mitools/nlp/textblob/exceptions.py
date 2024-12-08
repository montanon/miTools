MISSING_CORPUS_MESSAGE = """
Looks like you are missing some required data for this feature.

To download the necessary data, simply run

    python -m mitools.nlp.textblob.download_corpora

or use the NLTK downloader to download the missing data: http://nltk.org/data.html
"""


class TextBlobError(Exception):
    pass


class MissingCorpusError(TextBlobError):
    def __init__(self, message=MISSING_CORPUS_MESSAGE, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class DeprecationError(TextBlobError):
    pass


class TranslatorError(TextBlobError):
    pass


class NotTranslated(TranslatorError):
    pass


class FormatError(TextBlobError):
    pass
