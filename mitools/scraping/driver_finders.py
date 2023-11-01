from abc import ABC, abstractmethod

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class AbstractElementsFinder(ABC):

    @property
    @abstractmethod
    def by_type(self):
        pass

    def try_find_element(self, driver, identifier):
        try:
            self.find_element(driver, identifier)
        except NoSuchElementException as e:
            raise NoSuchElementException(
                f"""No element found with {self.by_type} = {identifier}\n\n
                Exception: {e}"""
            )
        
    def find_element(self, driver, identifier):
        return driver.find_element(self.by_type, identifier)

    def try_wait_and_find_element(self, driver, identifier, wait_time=2):
        try:
            self.wait_and_find_element(driver, identifier, wait_time)
        except TimeoutException as e:
            raise TimeoutException(
                f"""Element with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )
        
    def wait_and_find_element(self, driver, identifier, wait_time=2):
        return WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((self.by_type, identifier))
        )

    def try_find_elements(self, driver, identifier):
        try:
            self.find_elements(driver, identifier)
        except NoSuchElementException as e:
            raise NoSuchElementException(
                f"""No elements found with {self.by_type} = {identifier}\n\n
                Exception: {e}"""
            )
        
    def find_elements(self, driver, identifier):
        return driver.find_elements(self.by_type, identifier)

    def try_wait_and_find_elements(self, driver, identifier, wait_time=2):
        try:
            self.wait_and_find_elements(driver, identifier, wait_time)
        except TimeoutException as e:
            raise TimeoutException(
                f"""Elements with {self.by_type} = {identifier} not found in {wait_time} seconds\n\n
                Exception: {e}"""
            )
        
    def wait_and_find_elements(self, driver, identifier, wait_time=2):
        return WebDriverWait(driver, wait_time).until(
            EC.presence_of_all_elements_located((self.by_type, identifier))
        )


class IDFinder(AbstractElementsFinder):
    @property
    def by_type(self):
        return By.ID


class ClassNameFinder(AbstractElementsFinder):
    @property
    def by_type(self):
        return By.CLASS_NAME
    

class NameFinder(AbstractElementsFinder):
    @property
    def by_type(self):
        return By.NAME
    

class XPathFinder(AbstractElementsFinder):
    @property
    def by_type(self):
        return By.XPATH


class CSSSelectorFinder(AbstractElementsFinder):
    @property
    def by_type(self):
        return By.CSS_SELECTOR
    def convert_identifier(self, identifier):
        return '.' + identifier.replace(' ', '.')
    def try_find_element(self, driver, identifier):
        return super().try_find_element(driver, self.convert_identifier(identifier))
    def find_element(self, driver, identifier):
        return super().find_element(driver, self.convert_identifier(identifier))
    def try_wait_and_find_element(self, driver, identifier, wait_time=2):
        return super().try_wait_and_find_element(driver, self.convert_identifier(identifier), wait_time)
    def wait_and_find_element(self, driver, identifier, wait_time=2):
        return super().wait_and_find_element(driver, self.convert_identifier(identifier), wait_time)
    def try_find_elements(self, driver, identifier):
        return super().try_find_elements(driver, self.convert_identifier(identifier))
    def find_elements(self, driver, identifier):
        return super().find_elements(driver, self.convert_identifier(identifier))
    def try_wait_and_find_elements(self, driver, identifier, wait_time=2):
        return super().try_wait_and_find_elements(driver, self.convert_identifier(identifier), wait_time)
    def wait_and_find_elements(self, driver, identifier, wait_time=2):
        return super().wait_and_find_elements(driver, self.convert_identifier(identifier), wait_time)


class FinderFactory:

    def __init__(self):
        self._creators = {
            'id': IDFinder,
            'class': ClassNameFinder,
            'css': CSSSelectorFinder,
            'name': NameFinder,
            'xpath': XPathFinder
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
        return 'finder'
