from abc import ABC, abstractmethod

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class AbstractElementsWaiter(ABC):

    @property
    @abstractmethod
    def by_type(self):
        pass

    def try_wait_for_element(self, driver, identifier, wait_time=2):
        try:
            self.wait_for_element(driver, identifier, wait_time)
        except TimeoutException as e:
            raise TimeoutException(
                f"""Element with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )
        
    def wait_for_element(self, driver, identifier, wait_time=2):
        return WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((self.by_type, identifier))
            )
    

class IDWaiter(AbstractElementsWaiter):
    @property
    def by_type(self):
        return By.ID
    

class ClassNameWaiter(AbstractElementsWaiter):
    @property
    def by_type(self):
        return By.CLASS_NAME
    

class NameWaiter(AbstractElementsWaiter):
    @property
    def by_type(self):
        return By.NAME
    

class XPathWaiter(AbstractElementsWaiter):
    @property
    def by_type(self):
        return By.XPATH
    

class CSSSelectorWaiter(AbstractElementsWaiter):
    @property
    def by_type(self):
        return By.CSS_SELECTOR
    def convert_identifier(self, identifier):
        return '.' + identifier.replace(' ', '.')
    def try_wait_for_element(self, driver, identifier, wait_time=2):
        return super().try_wait_for_element(driver, self.convert_identifier(identifier), wait_time)
    def wait_for_element(self, driver, identifier, wait_time=2):
        return super().wait_for_element(driver, self.convert_identifier(identifier), wait_time)


class WaiterFactory:

    def __init__(self):
        self._creators = {
            'id': IDWaiter,
            'class': ClassNameWaiter,
            'css': CSSSelectorWaiter,
            'name': NameWaiter,
            'xpath': XPathWaiter
        }

    def get_object(self, selector):
        creator = self._creators.get(selector)
        if not creator:
            raise ValueError('Invalid selector')
        return creator()
    
    @property
    def creators(self):
        return list(self._creators.keys())

    @property
    def name(self):
        return 'waiter'
