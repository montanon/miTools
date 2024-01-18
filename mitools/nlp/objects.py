import pickle

from nltk.corpus import stopwords


class StopwordsManager:
    def __init__(self, language='english'):
        self.language = language
        self._words = set(stopwords.words(language))

    def add_stopword(self, word):
        self._words.add(word)

    def add_stopwords(self, words):
        self._words.update(words)

    def remove_stopword(self, word):
        self._words.discard(word)

    def remove_stopwords(self, words):
        self._words.difference_update(words)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    @property
    def words(self):
        return list(self._words)

