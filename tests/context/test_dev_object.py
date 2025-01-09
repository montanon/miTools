import json
import unittest
from pathlib import Path
from threading import Thread
from unittest import TestCase

from mitools.context import (
    Dev,
    get_dev_var,
    store_dev_var,
)


class TestDev(TestCase):
    def setUp(self):
        self.dev = Dev()  # Reference the singleton instance
        self.dev.clear_vars()  # Clear all variables before each test
        self.test_file = Path("./tests/.test_assets/dev_vars.json")

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_singleton_instance(self):
        dev1 = Dev()
        dev2 = Dev()
        self.assertIs(dev1, dev2)  # Both instances should point to the same object

    def test_store_and_get_var(self):
        self.dev.store_var("test_key", 123)
        self.assertEqual(self.dev.get_var("test_key"), 123)
        self.assertIsNone(self.dev.get_var("non_existent_key"))  # Key should not exist

    def test_store_var_invalid_key(self):
        with self.assertRaises(ValueError):
            self.dev.store_var(123, "value")  # Key must be a string

    def test_delete_var(self):
        self.dev.store_var("test_key", 123)
        self.dev.delete_var("test_key")
        with self.assertRaises(KeyError):
            self.dev.delete_var("test_key")  # Key should not exist

    def test_clear_vars(self):
        self.dev.store_var("key1", "value1")
        self.dev.store_var("key2", "value2")
        self.dev.clear_vars()
        self.assertEqual(len(self.dev.variables), 0)  # No variables should remain

    def test_save_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        with open(self.test_file) as f:
            dev_vars = json.load(f)
        dev_vars["key1"] == "value1"

    def test_load_variables(self):
        self.dev.store_var("key1", "value1")
        self.dev.save_variables(self.test_file)
        self.dev.load_variables(self.test_file)
        self.assertEqual(self.dev.get_var("key1"), "value1")

    def test_load_variables_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.dev.load_variables("non_existent_file.json")

    def test_store_var_using_global_function(self):
        store_dev_var("global_key", "global_value")
        self.assertEqual(get_dev_var("global_key"), "global_value")

    def test_thread_safety(self):
        def store_in_thread(dev, key, value):
            dev.store_var(key, value)

        thread1 = Thread(target=store_in_thread, args=(self.dev, "key1", "value1"))
        thread2 = Thread(target=store_in_thread, args=(self.dev, "key2", "value2"))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        # Both keys should exist in the dictionary
        self.assertEqual(self.dev.get_var("key1"), "value1")
        self.assertEqual(self.dev.get_var("key2"), "value2")

    def test_save_empty_variables(self):
        self.dev.clear_vars()
        with self.assertRaises(ValueError):
            self.dev.save_variables(self.test_file)


if __name__ == "__main__":
    unittest.main()
