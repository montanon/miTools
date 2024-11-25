import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumrequests import Chrome

from mitools.exceptions import (
    WebElementNotFoundError,
    WebScraperError,
    WebScraperTimeoutError,
)
from mitools.scraping.driver_checkers import (
    AbstractElementsPresenceChecker,
    ClassNamePresenceChecker,
    CSSSelectorPresenceChecker,
    IDPresenceChecker,
    NamePresenceChecker,
    PresenceCheckerFactory,
    XPathPresenceChecker,
)
from mitools.scraping.driver_finders import (
    AbstractElementsFinder,
    ClassNameFinder,
    CSSSelectorFinder,
    FinderFactory,
    IDFinder,
    NameFinder,
    XPathFinder,
)
from mitools.scraping.driver_waiters import (
    ClassNameWaiter,
    CSSSelectorWaiter,
    IDWaiter,
    NameWaiter,
    WaiterFactory,
    XPathWaiter,
)


class TestPresenceCheckers(TestCase):
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


class TestFinders(TestCase):
    def setUp(self):
        self.driver = MagicMock(spec=WebDriver)
        self.mock_element = MagicMock(spec=WebElement)
        self.mock_elements = [
            self.mock_element,
            self.mock_element,
        ]  # Simulate multiple elements
        self.wait_patch = patch("selenium.webdriver.support.ui.WebDriverWait").start()
        self.mock_wait = self.wait_patch.return_value
        self.addCleanup(patch.stopall)
        self.factory = FinderFactory()

    def test_find_element_success(self):
        self.driver.find_element.return_value = self.mock_element
        finder = IDFinder()
        result = finder.find_element(self.driver, "test-id")
        self.assertEqual(result, self.mock_element)
        self.driver.find_element.assert_called_with(By.ID, "test-id")

    def test_find_name_element_success(self):
        self.driver.find_element.return_value = self.mock_element
        finder = NameFinder()
        result = finder.find_element(self.driver, "test-id")
        self.assertEqual(result, self.mock_element)
        self.driver.find_element.assert_called_with(By.NAME, "test-id")

    def test_find_xpath_element_success(self):
        self.driver.find_element.return_value = self.mock_element
        finder = XPathFinder()
        result = finder.find_element(self.driver, "test-id")
        self.assertEqual(result, self.mock_element)
        self.driver.find_element.assert_called_with(By.XPATH, "test-id")

    def test_find_element_not_found(self):
        self.driver.find_element.side_effect = NoSuchElementException()
        finder = IDFinder()
        with self.assertRaises(WebElementNotFoundError):
            finder.try_find_element(self.driver, "nonexistent-id")

    def test_find_elements_success(self):
        self.driver.find_elements.return_value = self.mock_elements
        finder = ClassNameFinder()
        results = finder.find_elements(self.driver, "test-class")
        self.assertEqual(results, self.mock_elements)
        self.driver.find_elements.assert_called_with(By.CLASS_NAME, "test-class")

    def test_css_selector_conversion(self):
        self.driver.find_element.return_value = self.mock_element
        finder = CSSSelectorFinder()
        result = finder.find_element(self.driver, "test class")
        self.assertEqual(result, self.mock_element)
        self.driver.find_element.assert_called_with(By.CSS_SELECTOR, ".test.class")

    def test_factory_create_valid_finder(self):
        finder = self.factory.create("id")
        self.assertIsInstance(finder, IDFinder)

        finder = self.factory.create("css")
        self.assertIsInstance(finder, CSSSelectorFinder)

    def test_factory_create_invalid_finder(self):
        with self.assertRaises(ValueError):
            self.factory.create("invalid")

    def test_factory_register_custom_finder(self):
        class TagNameFinder(AbstractElementsFinder):
            @property
            def by_type(self):
                return By.TAG_NAME

        self.factory.register_finder("tag", TagNameFinder)
        finder = self.factory.create("tag")
        self.assertIsInstance(finder, TagNameFinder)

    def test_factory_creators_list(self):
        creators = self.factory.creators
        self.assertIn("id", creators)
        self.assertIn("css", creators)
        self.assertIn("xpath", creators)

    def test_convert_identifier_override(self):
        finder = CSSSelectorFinder()
        converted = finder.convert_identifier("test class")
        self.assertEqual(converted, ".test.class")

    def test_default_convert_identifier(self):
        finder = IDFinder()
        self.assertEqual(finder.convert_identifier("test-id"), "test-id")


class MockWebDriver(WebDriver):
    def __init__(self, elements=None):
        self.elements = elements or {}

    def find_element(self, by, value):
        print(by, value)
        if (by, value) in self.elements:
            return self.elements[(by, value)]
        raise TimeoutException("Element not found")

    def find_elements(self, by, value):
        return [
            element
            for (b, v), element in self.elements.items()
            if b == by and v == value
        ]


class TestWaiters(TestCase):
    def setUp(self):
        self.mock_elements = {
            (By.ID, "test-id"): "Element with ID",
            (By.CLASS_NAME, "test-class"): "Element with class name",
            (By.NAME, "test-name"): "Element with name",
            (By.XPATH, "//div[@class='test']"): "Element with XPath",
            (By.CSS_SELECTOR, "test class"): "Element with CSS",
        }
        self.driver = MockWebDriver(elements=self.mock_elements)

    def test_wait_for_element_success(self):
        waiter = IDWaiter()
        result = waiter.wait_for_element(self.driver, "test-id", wait_time=1)
        self.assertEqual(result, "Element with ID")

    def test_wait_for_element_not_found(self):
        waiter = IDWaiter()
        with self.assertRaises(TimeoutException):
            waiter.wait_for_element(self.driver, "nonexistent-id", wait_time=1)

    def test_try_wait_for_element_success(self):
        waiter = NameWaiter()
        try:
            waiter.try_wait_for_element(self.driver, "test-name", wait_time=1)
        except WebScraperTimeoutError:
            self.fail("try_wait_for_element raised WebScraperTimeoutError unexpectedly")

    def test_try_wait_for_element_timeout(self):
        waiter = XPathWaiter()
        with self.assertRaises(WebScraperTimeoutError):
            waiter.try_wait_for_element(
                self.driver, "//div[@class='nonexistent']", wait_time=1
            )

    def test_css_selector_conversion(self):
        waiter = CSSSelectorWaiter()
        self.assertEqual(waiter.convert_identifier("test class"), ".test.class")

    def test_wait_for_element_class_name_success(self):
        waiter = ClassNameWaiter()
        result = waiter.wait_for_element(self.driver, "test-class", wait_time=1)
        self.assertEqual(result, "Element with class name")

    def test_wait_for_element_css_success(self):
        waiter = CSSSelectorWaiter()
        result = waiter.wait_for_element(self.driver, "test class", wait_time=1)
        self.assertEqual(result, "Element with CSS")

    def test_wait_for_element_css_not_found(self):
        waiter = CSSSelectorWaiter()
        with self.assertRaises(TimeoutException):
            waiter.wait_for_element(self.driver, "nonexistent class", wait_time=1)

    def test_factory_create_valid_waiter(self):
        factory = WaiterFactory()
        waiter = factory.create("id")
        self.assertIsInstance(waiter, IDWaiter)

        waiter = factory.create("css")
        self.assertIsInstance(waiter, CSSSelectorWaiter)

    def test_factory_create_invalid_waiter(self):
        factory = WaiterFactory()
        with self.assertRaises(WebScraperError):
            factory.create("invalid")

    def test_factory_creators_list(self):
        factory = WaiterFactory()
        creators = factory.creators
        self.assertIn("id", creators)
        self.assertIn("class", creators)
        self.assertIn("xpath", creators)

    def test_factory_create_and_use_waiter(self):
        factory = WaiterFactory()
        waiter = factory.create("xpath")
        result = waiter.wait_for_element(
            self.driver, "//div[@class='test']", wait_time=1
        )
        self.assertEqual(result, "Element with XPath")

    def test_default_convert_identifier(self):
        waiter = IDWaiter()
        self.assertEqual(waiter.convert_identifier("test-id"), "test-id")


if __name__ == "__main__":
    unittest.main()
