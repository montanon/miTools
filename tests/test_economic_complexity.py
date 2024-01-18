import unittest

from mitools.economic_complexity import StringMapper


class TestStringConverter(unittest.TestCase):
    
    def setUp(self):
        self.relations = {'pretty1': 'ugly1', 'pretty2': 'ugly2'}
        self.converter = StringMapper(self.relations)

    def test_add_relation(self):
        self.converter.add_relation('pretty3', 'ugly3')
        self.assertEqual(self.converter.prettify_str('ugly3'), 'pretty3')
        self.assertEqual(self.converter.uglify_str('pretty3'), 'ugly3')

    def test_add_relation_case_insensitive(self):
        converter = StringMapper(self.relations, case_sensitive=False)
        converter.add_relation('Pretty4', 'Ugly4')
        self.assertEqual(converter.prettify_str('ugly4'), 'pretty4')
        self.assertEqual(converter.uglify_str('pretty4'), 'ugly4')

    def test_add_relation_duplicate(self):
        with self.assertRaises(ValueError):
            self.converter.add_relation('pretty1', 'ugly3')

    def test_prettify_str(self):
        self.assertEqual(self.converter.prettify_str('ugly1'), 'pretty1')

    def test_prettify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.converter.prettify_str('ugly3')

    def test_prettify_strs(self):
        self.assertEqual(self.converter.prettify_strs(['ugly1', 'ugly2']), ['pretty1', 'pretty2'])

    def test_uglify_str(self):
        self.assertEqual(self.converter.uglify_str('pretty1'), 'ugly1')

    def test_uglify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.converter.uglify_str('pretty3')

    def test_uglify_strs(self):
        self.assertEqual(self.converter.uglify_strs(['pretty1', 'pretty2']), ['ugly1', 'ugly2'])

if __name__ == '__main__':
    unittest.main()