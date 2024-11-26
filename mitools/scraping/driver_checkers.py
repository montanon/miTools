from abc import ABC, abstractmethod
from typing import Dict, List, Type

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from mitools.exceptions import ArgumentValueError


class AbstractElementsPresenceChecker(ABC):
    @property
    @abstractmethod
    def by_type(self) -> By:
        pass

    def is_element_present(self, driver: WebDriver, identifier: str) -> bool:
        elements = driver.find_elements(self.by_type, identifier)
        return len(elements) > 0

    def wait_and_is_element_present(self, driver, identifier, wait_time=2):
        try:
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((self.by_type, identifier))
            )
            return True
        except TimeoutException:
            return False
        except Exception as e:
            raise e


class IDPresenceChecker(AbstractElementsPresenceChecker):
    @property
    def by_type(self):
        return By.ID


class ClassNamePresenceChecker(AbstractElementsPresenceChecker):
    @property
    def by_type(self):
        return By.CLASS_NAME


class NamePresenceChecker(AbstractElementsPresenceChecker):
    @property
    def by_type(self):
        return By.NAME


class XPathPresenceChecker(AbstractElementsPresenceChecker):
    @property
    def by_type(self):
        return By.XPATH


class CSSSelectorPresenceChecker(AbstractElementsPresenceChecker):
    @property
    def by_type(self):
        return By.CSS_SELECTOR

    def convert_identifier(self, identifier):
        return "." + identifier.replace(" ", ".")

    def is_element_present(self, driver, identifier):
        return super().is_element_present(driver, self.convert_identifier(identifier))

    def wait_and_is_element_present(self, driver, identifier, wait_time=2):
        return super().wait_and_is_element_present(
            driver, self.convert_identifier(identifier), wait_time
        )


class PresenceCheckerFactory:
    def __init__(self):
        self._creators: Dict[str, Type[AbstractElementsPresenceChecker]] = {
            "id": IDPresenceChecker,
            "class": ClassNamePresenceChecker,
            "css": CSSSelectorPresenceChecker,
            "name": NamePresenceChecker,
            "xpath": XPathPresenceChecker,
        }

    def create(self, selector: str) -> AbstractElementsPresenceChecker:
        creator = self._creators.get(selector)
        if not creator:
            raise ArgumentValueError(f"Invalid selector type: {selector}")
        return creator()

    def register_checker(
        self, selector: str, checker_cls: Type[AbstractElementsPresenceChecker]
    ):
        self._creators[selector] = checker_cls

    @property
    def creators(self) -> List[str]:
        return list(self._creators.keys())

    @property
    def name(self) -> str:
        return "presence_checker"
