from abc import ABC, abstractmethod
from typing import List

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mitools.exceptions import WebScraperError, WebScraperTimeoutError


class AbstractElementsWaiter(ABC):
    @property
    @abstractmethod
    def by_type(self):
        pass

    def convert_identifier(self, identifier: str) -> str:
        return identifier

    def try_wait_for_element(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> None:
        identifier = self.convert_identifier(identifier)
        try:
            self.wait_for_element(driver, identifier, wait_time)
        except TimeoutException as e:
            raise WebScraperTimeoutError(
                f"""Element with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )

    def wait_for_element(
        self, driver: WebDriver, identifier: str, wait_time: int = 2
    ) -> None:
        return WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((self.by_type, identifier))
        )


class IDWaiter(AbstractElementsWaiter):
    @property
    def by_type(self) -> By:
        return By.ID


class ClassNameWaiter(AbstractElementsWaiter):
    @property
    def by_type(self) -> By:
        return By.CLASS_NAME


class NameWaiter(AbstractElementsWaiter):
    @property
    def by_type(self) -> By:
        return By.NAME


class XPathWaiter(AbstractElementsWaiter):
    @property
    def by_type(self) -> By:
        return By.XPATH


class CSSSelectorWaiter(AbstractElementsWaiter):
    @property
    def by_type(self) -> By:
        return By.CSS_SELECTOR

    def convert_identifier(self, identifier: str) -> str:
        return "." + identifier.replace(" ", ".")


class WaiterFactory:
    def __init__(self):
        self._creators = {
            "id": IDWaiter,
            "class": ClassNameWaiter,
            "css": CSSSelectorWaiter,
            "name": NameWaiter,
            "xpath": XPathWaiter,
        }

    def create_waiter(self, selector: str) -> AbstractElementsWaiter:
        creator = self._creators.get(selector)
        if not creator:
            raise WebScraperError("Invalid selector")
        return creator()

    @property
    def creators(self) -> List[str]:
        return list(self._creators.keys())

    @property
    def name(self) -> str:
        return "waiter"
