import unittest

from mitools.economic_complexity import StringMapper


class TestStringMapper(unittest.TestCase):

    def setUp(self):
        self.relations = {'pretty1': 'ugly1', 'pretty2': 'ugly2'}
        self.mapper = StringMapper(self.relations)

    def test_add_relation(self):
        self.mapper.add_relation('pretty3', 'ugly3')
        self.assertEqual(self.mapper.prettify_str('ugly3'), 'pretty3')
        self.assertEqual(self.mapper.uglify_str('pretty3'), 'ugly3')

    def test_add_relation_case_insensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=False)
        mapper.add_relation('Pretty4', 'Ugly4')
        self.assertEqual(mapper.prettify_str('ugly4'), 'pretty4')
        self.assertEqual(mapper.uglify_str('pretty4'), 'ugly4')
        self.assertEqual(mapper.prettify_str('Ugly4'), 'pretty4')
        self.assertEqual(mapper.uglify_str('Pretty4'), 'ugly4')

    def test_add_relation_case_sensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=True)
        mapper.add_relation('Pretty4', 'Ugly4')
        self.assertEqual(mapper.prettify_str('Ugly4'), 'Pretty4')
        self.assertEqual(mapper.uglify_str('Pretty4'), 'Ugly4')
        self.assertNotEqual(mapper.prettify_str('Ugly4'), 'pretty4')
        self.assertNotEqual(mapper.uglify_str('Pretty4'), 'ugly4')

    def test_add_relation_duplicate(self):
        with self.assertRaises(ValueError):
            self.mapper.add_relation('pretty1', 'ugly3')

    def test_prettify_str(self):
        self.assertEqual(self.mapper.prettify_str('ugly1'), 'pretty1')

    def test_prettify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.prettify_str('ugly3')

    def test_prettify_strs(self):
        self.assertEqual(self.mapper.prettify_strs(['ugly1', 'ugly2']), ['pretty1', 'pretty2'])

    def test_uglify_str(self):
        self.assertEqual(self.mapper.uglify_str('pretty1'), 'ugly1')

    def test_uglify_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.uglify_str('pretty3')

    def test_uglify_strs(self):
        self.assertEqual(self.mapper.uglify_strs(['pretty1', 'pretty2']), ['ugly1', 'ugly2'])

    def test_remap_str(self):
        self.assertEqual(self.mapper.remap_str('pretty1'), 'ugly1')
        self.assertEqual(self.mapper.remap_str('ugly1'), 'pretty1')

    def test_remap_str_not_found(self):
        with self.assertRaises(ValueError):
            self.mapper.remap_str('pretty3')

    def test_remap_strs(self):
        self.assertEqual(self.mapper.remap_strs(['pretty1', 'pretty2']), ['ugly1', 'ugly2'])
        self.assertEqual(self.mapper.remap_strs(['ugly1', 'ugly2']), ['pretty1', 'pretty2'])

    def test_remap_strs_mixed(self):
        with self.assertRaises(ValueError):
            self.mapper.remap_strs(['pretty1', 'ugly2'])

    def test_is_pretty(self):
        self.assertTrue(self.mapper.is_pretty('pretty1'))
        self.assertFalse(self.mapper.is_pretty('ugly1'))

    def test_is_ugly(self):
        self.assertTrue(self.mapper.is_ugly('ugly1'))
        self.assertFalse(self.mapper.is_ugly('pretty1'))

    def test_case_insensitive(self):
        mapper = StringMapper(self.relations, case_sensitive=False)
        self.assertTrue(mapper.is_pretty('PRETTY1'))
        self.assertTrue(mapper.is_ugly('UGLY1'))
        self.assertEqual(mapper.prettify_str('UGLY1'), 'pretty1')
        self.assertEqual(mapper.uglify_str('PRETTY1'), 'ugly1')

if __name__ == '__main__':
    unittest.main()