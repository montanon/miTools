import unittest
from pathlib import Path

from mitools.nlp.objects import Lexicon, Morphology


class TestMorphology(unittest.TestCase):
    def setUp(self):
        self.lexicon = {
            "play": "VB",
            "plays": "VBZ",
            "playing": "VBG",
            "played": "VBD",
            "player": "NN",
            "players": "NNS",
            "quick": "JJ",
            "quickly": "RB",
            "happy": "JJ",
            "unhappy": "JJ",
            "happiness": "NN",
            "cat": "NN",
            "cats": "NNS",
        }
        self.morphology = Morphology(lexicon=self.lexicon)
        self.test_rules = [
            ["NN", "s", "fhassuf", "1", "NNS", "x"],  # Plural rule
            ["ly", "hassuf", "2", "RB", "x"],  # Adverb rule
            ["un", "deletepref", "2", "JJ", "x"],  # Unhappy -> happy rule
            ["NN", "ing", "fhassuf", "3", "VBG", "x"],  # Gerund rule
        ]
        # for rule in self.test_rules:
        #    self.morphology.append(*rule)

    def test_initialization(self):
        self.assertIsInstance(self.morphology, Morphology)
        self.assertEqual(len(self.morphology), len(self.test_rules))
        self.assertEqual(self.morphology.lexicon, self.lexicon)

    def test_load_from_file(self):
        morph = Morphology(path="mitools/nlp/en/en-morphology.txt")
        morph.load()
        print(morph.morp_operations)
        self.assertGreater(len(morph), 0)
        rules_str = str(morph)
        self.assertIn("NN s fhassuf 1 NNS x", rules_str)
        self.assertIn("ly hassuf 2 RB x", rules_str)

    def test_apply_plural_rule(self):
        token = ["cat", "NN"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "NN")  # Should stay NN as is
        token = ["cats", "NN"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "NNS")  # Should transform to NNS

    def test_apply_adverb_rule(self):
        token = ["quickly", "JJ"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "RB")  # Should transform to RB

    def test_apply_prefix_rule(self):
        token = ["unhappy", "JJ"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "JJ")  # Should stay JJ
        token = ["happy", "JJ"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "JJ")

    def test_apply_gerund_rule(self):
        token = ["playing", "NN"]
        result = self.morphology.apply(token)
        self.assertEqual(result[1], "VBG")  # Should transform to VBG

    def test_apply_with_context(self):
        token = ["play", "VB"]
        prev = ["I", "PRP"]
        next_token = ["the", "DT"]
        result = self.morphology.apply(token, previous=prev, next=next_token)
        self.assertEqual(result[1], "VB")  # Should maintain VB tag

    def test_insert_rule(self):
        initial_length = len(self.morphology)
        self.morphology.insert(0, "NN", "er", "fhassuf", "2", "JJR")
        self.assertEqual(len(self.morphology), initial_length + 1)
        self.assertEqual(self.morphology[0], ["NN", "er", "fhassuf", "2", "JJR", "x"])

    def test_append_rule(self):
        initial_length = len(self.morphology)
        self.morphology.append("JJ", "est", "fhassuf", "3", "JJS")
        self.assertEqual(len(self.morphology), initial_length + 1)
        self.assertEqual(self.morphology[-1], ["JJ", "est", "fhassuf", "3", "JJS", "x"])

    def test_extend_rules(self):
        initial_length = len(self.morphology)
        new_rules = [
            ("NN", "able", "fhassuf", "4", "JJ"),
            ("JJ", "ness", "fhassuf", "4", "NN"),
        ]
        self.morphology.extend(new_rules)
        self.assertEqual(len(self.morphology), initial_length + 2)

    def test_invalid_rule_application(self):
        token = ["", ""]
        result = self.morphology.apply(token)
        self.assertEqual(result, token)
        token = [None, None]
        result = self.morphology.apply(token)
        self.assertEqual(result, token)


if __name__ == "__main__":
    unittest.main()
