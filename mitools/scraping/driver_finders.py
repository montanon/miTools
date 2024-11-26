from abc import ABC, abstractmethod
from typing import List

from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver, WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mitools.exceptions import WebElementNotFoundError, WebScraperTimeoutError


class AbstractElementsFinder(ABC):
    @property
    @abstractmethod
    def by_type(self):
        pass

    def try_find_element(self, driver: WebDriver, identifier: str) -> WebElement:
        try:
            self.find_element(driver, identifier)
        except NoSuchElementException as e:
            raise WebElementNotFoundError(
                f"""No element found with {self.by_type} = {identifier}\n\n
                Exception: {e}"""
            )

    def find_element(self, driver: WebDriver, identifier: str) -> WebElement:
        identifier = self.convert_identifier(identifier)
        return driver.find_element(self.by_type, identifier)

    def convert_identifier(self, identifier: str) -> str:
        return identifier

    def try_wait_and_find_element(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> WebElement:
        try:
            self.wait_and_find_element(driver, identifier, wait_time)
        except TimeoutException as e:
            raise WebScraperTimeoutError(
                f"""Element with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )

    def wait_and_find_element(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> WebElement:
        return WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((self.by_type, identifier))
        )

    def try_find_elements(self, driver: WebDriver, identifier: str) -> List[WebElement]:
        try:
            self.find_elements(driver, identifier)
        except NoSuchElementException as e:
            raise WebElementNotFoundError(
                f"""No elements found with {self.by_type} = {identifier}\n\n
                Exception: {e}"""
            )

    def find_elements(self, driver: WebDriver, identifier: str) -> List[WebElement]:
        return driver.find_elements(self.by_type, identifier)

    def try_wait_and_find_elements(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> List[WebElement]:
        try:
            self.wait_and_find_elements(driver, identifier, wait_time)
        except TimeoutException as e:
            raise WebScraperTimeoutError(
                f"""Elements with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )

    def wait_and_find_elements(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> List[WebElement]:
        return WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((self.by_type, identifier))
        )


class IDFinder(AbstractElementsFinder):
    @property
    def by_type(self) -> By:
        return By.ID


class ClassNameFinder(AbstractElementsFinder):
    @property
    def by_type(self) -> By:
        return By.CLASS_NAME


class NameFinder(AbstractElementsFinder):
    @property
    def by_type(self) -> By:
        return By.NAME


class XPathFinder(AbstractElementsFinder):
    @property
    def by_type(self) -> By:
        return By.XPATH


class CSSSelectorFinder(AbstractElementsFinder):
    @property
    def by_type(self) -> By:
        return By.CSS_SELECTOR

    def convert_identifier(self, identifier):
        return "." + identifier.replace(" ", ".")


class FinderFactory:
    def __init__(self):
        self._creators = {
            "id": IDFinder,
            "class": ClassNameFinder,
            "css": CSSSelectorFinder,
            "name": NameFinder,
            "xpath": XPathFinder,
        }

    def create(self, selector: str) -> AbstractElementsFinder:
        creator = self._creators.get(selector)
        if not creator:
            raise ValueError("Invalid selector")
        return creator()

    def register_finder(self, selector: str, finder_cls: AbstractElementsFinder):
        self._creators[selector] = finder_cls

    @property
    def creators(self) -> List[str]:
        return list(self._creators.keys())

    @property
    def name(self) -> str:
        return "finder"
