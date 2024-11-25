import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mitools.exceptions import WebScraperError, WebScraperTimeoutError
from mitools.scraping.driver_checkers import (
    AbstractElementsPresenceChecker,
    ClassNamePresenceChecker,
    CSSSelectorPresenceChecker,
    IDPresenceChecker,
    NamePresenceChecker,
    PresenceCheckerFactory,
    XPathPresenceChecker,
)


class TestPresenceCheckers(unittest.TestCase):
    def setUp(self):
        self.driver = MagicMock(spec=WebDriver)
        self.mock_elements = [MagicMock(), MagicMock()]  # Simulate multiple elements
        self.wait_mock = patch("selenium.webdriver.support.ui.WebDriverWait").start()
        self.addCleanup(patch.stopall)
        self.factory = PresenceCheckerFactory()

    def test_is_element_present_id(self):
        checker = IDPresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "test-id"))

    def test_is_element_present_class_name(self):
        checker = ClassNamePresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "test-class"))

    def test_is_element_present_name(self):
        checker = NamePresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "test-name"))

    def test_is_element_present_xpath(self):
        checker = XPathPresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "//div[@class='test']"))

    def test_is_element_present_css(self):
        checker = CSSSelectorPresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "test css class"))

    def test_is_element_not_present(self):
        checker = IDPresenceChecker()
        self.driver.find_elements.return_value = []
        self.assertFalse(checker.is_element_present(self.driver, "nonexistent-id"))

    def test_wait_and_is_element_present(self):
        checker = IDPresenceChecker()
        self.wait_mock.return_value.until.return_value = True
        self.assertTrue(checker.wait_and_is_element_present(self.driver, "test-id"))

    def test_factory_creates_checker(self):
        checker = self.factory.create("id")
        self.assertIsInstance(checker, IDPresenceChecker)

        checker = self.factory.create("css")
        self.assertIsInstance(checker, CSSSelectorPresenceChecker)

    def test_factory_creators_list(self):
        creators = self.factory.creators
        self.assertIn("id", creators)
        self.assertIn("css", creators)

    def test_factory_register_checker(self):
        class CustomChecker(AbstractElementsPresenceChecker):
            @property
            def by_type(self):
                return By.TAG_NAME

        self.factory.register_checker("tag", CustomChecker)
        checker = self.factory.create("tag")
        self.assertIsInstance(checker, CustomChecker)

    def test_invalid_selector_raises_error(self):
        with self.assertRaises(
            Exception
        ):  # Replace with ArgumentValueError if imported
            self.factory.create("invalid")

    def test_css_selector_conversion(self):
        checker = CSSSelectorPresenceChecker()
        self.driver.find_elements.return_value = self.mock_elements
        self.assertTrue(checker.is_element_present(self.driver, "test class"))
        self.driver.find_elements.assert_called_with(By.CSS_SELECTOR, ".test.class")


if __name__ == "__main__":
    unittest.main()