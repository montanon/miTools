import unittest
from unittest import TestCase

from mitools.context import Context, ContextVar

VARIABLE = ContextVar("VARIABLE", 0)


class TestContextVars(TestCase):
    # Ensuring that the test does not modify variables outside the tests.
    ctx = Context()

    def setUp(self):
        TestContextVars.ctx.__enter__()

    def tearDown(self):
        TestContextVars.ctx.__exit__()

    def test_initial_value_is_set(self):
        _TMP = ContextVar("_TMP", 5)
        self.assertEqual(_TMP.value, 5)

    def test_multiple_creation_ignored(self):
        _TMP2 = ContextVar("_TMP2", 1)
        _TMP2 = ContextVar("_TMP2", 2)
        self.assertEqual(_TMP2.value, 1)

    def test_new_var_inside_context(self):
        # Creating a _new_ variable inside a context should not have any effect on its scope (?)
        with Context(VARIABLE=1):
            _TMP3 = ContextVar("_TMP3", 1)
        _TMP3 = ContextVar("_TMP3", 2)
        self.assertEqual(_TMP3.value, 1)

    def test_value_accross_modules(self):
        # Mocking module import by invoking the code but not in our globals().
        exec('from mitools.context import ContextVar;C = ContextVar("C", 13)', {})  # pylint:disable=exec-used
        # It should not matter that the first creation was in another module.
        C = ContextVar("C", 0)
        self.assertEqual(C.value, 13)

    def test_assignment_across_modules(self):
        B = ContextVar("B", 1)
        # local assignment
        B.value = 2
        self.assertEqual(B.value, 2)
        # Assignment in another module.
        exec(
            'from mitools.context import ContextVar;B = ContextVar("B", 0);B.value = 3;',
            {},
        )  # pylint:disable=exec-used
        # Assignment in another module should affect this one as well.
        self.assertEqual(B.value, 3)

    def test_context_assignment(self):
        with Context(VARIABLE=1):
            self.assertEqual(VARIABLE.value, 1)
        self.assertEqual(VARIABLE.value, 0)

    def test_unknown_param_to_context(self):
        with self.assertRaises(KeyError):
            with Context(SOMETHING_ELSE=1):
                pass

    def test_inside_context_assignment(self):
        with Context(VARIABLE=4):
            # What you can and cannot do inside a context.
            # 1. This type of statement has no effect.
            VARIABLE = ContextVar("VARIABLE", 0)
            self.assertTrue(
                VARIABLE >= 4,
                "ContextVars inside contextmanager may not set a new value",
            )

            # 2. The call syntax however has a local effect.
            VARIABLE.value = 13
            self.assertTrue(
                VARIABLE.value == 13,
                "Call syntax however works inside a contextmanager.",
            )

        # Related to 2. above. Note that VARIABLE is back to 0 again as expected.
        self.assertEqual(VARIABLE.value, 0)

    def test_new_var_inside_context_other_module(self):
        with Context(VARIABLE=1):
            _NEW2 = ContextVar("_NEW2", 0)
        _NEW2 = ContextVar("_NEW2", 1)
        self.assertEqual(_NEW2.value, 0)

        code = """\
from mitools.context import Context, ContextVar
with Context(VARIABLE=1):
  _NEW3 = ContextVar("_NEW3", 0)"""
        exec(code, {})  # pylint:disable=exec-used
        # While _NEW3 was created in an outside scope it should still work the same as above.
        _NEW3 = ContextVar("_NEW3", 1)
        self.assertEqual(_NEW3.value, 0)

    def test_nested_context(self):
        with Context(VARIABLE=1):
            with Context(VARIABLE=2):
                with Context(VARIABLE=3):
                    self.assertEqual(VARIABLE.value, 3)
                self.assertEqual(VARIABLE.value, 2)
            self.assertEqual(VARIABLE.value, 1)
        self.assertEqual(VARIABLE.value, 0)

    def test_decorator(self):
        @Context(VARIABLE=1, DEBUG=4)
        def test():
            self.assertEqual(VARIABLE.value, 1)

        self.assertEqual(VARIABLE.value, 0)
        test()
        self.assertEqual(VARIABLE.value, 0)

    def test_context_exit_reverts_updated_values(self):
        D = ContextVar("D", 1)
        D.value = 2
        with Context(D=3):
            ...
        assert (
            D.value == 2
        ), f"Expected D to be 2, but was {D.value}. Indicates that Context.__exit__ did not restore to the correct value."


if __name__ == "__main__":
    unittest.main()
