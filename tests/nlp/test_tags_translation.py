import unittest
from unittest import TestCase

from nltk.corpus import wordnet

from mitools.nlp.tags_translator import (
    ADJ,
    ADV,
    CONJ,
    DET,
    INTJ,
    NOUN,
    NUM,
    PREP,
    PRON,
    PRT,
    PUNC,
    VERB,
    X,
    translate_tag,
)


class TestTranslateTag(TestCase):
    def test_return_X_if_input_X(self):
        # If input tag == X, should return X regardless of formats
        for sf in ["penn", "universal", "wordnet", "nltk"]:
            for tf in ["penn", "universal", "wordnet", "nltk"]:
                with self.subTest(source=sf, target=tf):
                    self.assertEqual(translate_tag("X", sf, tf), "X")

    def test_direct_penn_universal(self):
        penn_universal_samples = {
            "NN": NOUN,
            "NNS": NOUN,
            "NNP": NOUN,
            "VB": VERB,
            "VBD": VERB,
            "VBP": VERB,
            "JJ": ADJ,
            "JJR": ADJ,
            "RB": ADV,
            "RBR": ADV,
            "RBS": ADV,
            "PRP": PRON,
            "DT": DET,
            "IN": PREP,
            "CD": NUM,
            "CC": CONJ,
            "UH": INTJ,
            "RP": PRT,
            ".": PUNC,
            ",": PUNC,
        }
        for penn_tag, expected_universal in penn_universal_samples.items():
            with self.subTest(penn_tag=penn_tag):
                self.assertEqual(
                    translate_tag(penn_tag, "penn", "universal"), expected_universal
                )

    def test_direct_universal_penn(self):
        universal_penn_samples = {
            NOUN: "NN",
            VERB: "VB",
            ADJ: "JJ",
            ADV: "RB",
            PRON: "PR",
            DET: "DT",
            PREP: "PP",
            NUM: "NO",
            CONJ: "CJ",
            INTJ: "UH",
            PRT: "PT",
            PUNC: ".",
            X: "X",
        }
        for universal_tag, expected_penn in universal_penn_samples.items():
            with self.subTest(universal_tag=universal_tag):
                self.assertEqual(
                    translate_tag(universal_tag, "universal", "penn"), expected_penn
                )

    def test_direct_wordnet_universal(self):
        wordnet_universal_samples = {
            wordnet.ADJ: ADJ,
            wordnet.ADJ_SAT: ADJ,
            wordnet.ADV: ADV,
            wordnet.NOUN: NOUN,
            wordnet.VERB: VERB,
        }
        for wn_tag, expected_universal in wordnet_universal_samples.items():
            with self.subTest(wn_tag=wn_tag):
                self.assertEqual(
                    translate_tag(wn_tag, "wordnet", "universal"), expected_universal
                )

    def test_direct_universal_wordnet(self):
        universal_wordnet_samples = {
            ADJ: wordnet.ADJ,
            ADV: wordnet.ADV,
            NOUN: wordnet.NOUN,
            VERB: wordnet.VERB,
        }
        for universal_tag, expected_wn in universal_wordnet_samples.items():
            with self.subTest(universal_tag=universal_tag):
                self.assertEqual(
                    translate_tag(universal_tag, "universal", "wordnet"), expected_wn
                )

    def test_direct_nltk_universal(self):
        nltk_universal_samples = {
            "J": ADJ,
            "V": VERB,
            "N": NOUN,
            "R": ADV,
        }
        for nltk_tag, expected_universal in nltk_universal_samples.items():
            with self.subTest(nltk_tag=nltk_tag):
                self.assertEqual(
                    translate_tag(nltk_tag, "nltk", "universal"), expected_universal
                )

    def test_direct_universal_nltk(self):
        # UNIVERSAL_TO_NLTK (inferred from NLTK_TO_UNIVERSAL)
        universal_nltk_samples = {
            ADJ: "J",
            VERB: "V",
            NOUN: "N",
            ADV: "R",
        }
        for universal_tag, expected_nltk in universal_nltk_samples.items():
            with self.subTest(universal_tag=universal_tag):
                self.assertEqual(
                    translate_tag(universal_tag, "universal", "nltk"), expected_nltk
                )

    def test_indirect_penn_wordnet(self):
        self.assertEqual(translate_tag("NN", "penn", "wordnet"), wordnet.NOUN)
        self.assertEqual(translate_tag("VB", "penn", "wordnet"), wordnet.VERB)
        self.assertEqual(translate_tag("JJ", "penn", "wordnet"), wordnet.ADJ)

    def test_indirect_wordnet_penn(self):
        self.assertEqual(translate_tag(wordnet.NOUN, "wordnet", "penn"), "NN")
        self.assertEqual(translate_tag(wordnet.VERB, "wordnet", "penn"), "VB")
        self.assertEqual(translate_tag(wordnet.ADJ, "wordnet", "penn"), "JJ")

    def test_indirect_penn_nltk(self):
        self.assertEqual(translate_tag("NN", "penn", "nltk"), "N")
        self.assertEqual(translate_tag("VB", "penn", "nltk"), "V")
        self.assertEqual(translate_tag("JJ", "penn", "nltk"), "J")

    def test_indirect_nltk_penn(self):
        self.assertEqual(translate_tag("N", "nltk", "penn"), "NN")
        self.assertEqual(translate_tag("V", "nltk", "penn"), "VB")
        self.assertEqual(translate_tag("J", "nltk", "penn"), "JJ")

    def test_indirect_nltk_wordnet(self):
        self.assertEqual(translate_tag("N", "nltk", "wordnet"), wordnet.NOUN)
        self.assertEqual(translate_tag("V", "nltk", "wordnet"), wordnet.VERB)
        self.assertEqual(translate_tag("J", "nltk", "wordnet"), wordnet.ADJ)

    def test_indirect_wordnet_nltk(self):
        self.assertEqual(translate_tag(wordnet.NOUN, "wordnet", "nltk"), "N")
        self.assertEqual(translate_tag(wordnet.VERB, "wordnet", "nltk"), "V")
        self.assertEqual(translate_tag(wordnet.ADJ, "wordnet", "nltk"), "J")

    def test_unknown_tags(self):
        self.assertEqual(translate_tag("UNKNOWN", "penn", "universal"), "X")
        self.assertEqual(translate_tag("XYZ", "universal", "penn"), "X")
        self.assertEqual(translate_tag("Z", "nltk", "wordnet"), "X")

    def test_same_format_translation(self):
        self.assertEqual(translate_tag(NOUN, "universal", "universal"), NOUN)
        self.assertEqual(translate_tag("NN", "penn", "penn"), "NN")
        self.assertEqual(translate_tag("N", "nltk", "nltk"), "N")
        self.assertEqual(
            translate_tag(wordnet.NOUN, "wordnet", "wordnet"), wordnet.NOUN
        )
        self.assertEqual(translate_tag(X, "penn", "penn"), X)

    def test_punctuation_cross_translations(self):
        self.assertEqual(translate_tag(".", "penn", "universal"), PUNC)
        self.assertEqual(translate_tag(PUNC, "universal", "penn"), ".")
        self.assertEqual(translate_tag(PUNC, "universal", "nltk"), "X")

    def test_multi_step_fail(self):
        self.assertEqual(translate_tag("IN", "penn", "nltk"), "X")
        self.assertEqual(translate_tag("Xsomething", "nltk", "wordnet"), "X")


if __name__ == "__main__":
    unittest.main()
