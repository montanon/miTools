from abc import ABC, abstractmethod

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class AbstractElementsPresenceChecker(ABC):

    @property
    @abstractmethod
    def by_type(self):
        pass

    def is_element_present(self, driver, identifier):
        elements = driver.find_elements(self.by_type, identifier)
        return len(elements) > 0
    
    def wait_and_is_element_present(self, driver, identifier, wait_time=2):
        try:
            WebDriverWait(driver, wait_time).until(
                EC.presence_of_element_located((self.by_type, identifier))
            )
            return True
        except Exception:
            return False
        

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
        return '.' + identifier.replace(' ', '.')
    def is_element_present(self, driver, identifier):
        return super().is_element_present(driver, self.convert_identifier(identifier))
    def wait_and_is_element_present(self, driver, identifier, wait_time=2):
        return super().wait_and_is_element_present(driver, self.convert_identifier(identifier), wait_time)


class PresenceCheckerFactory:

    def __init__(self):
        self._creators = {
            'id': IDPresenceChecker,
            'class': ClassNamePresenceChecker,
            'css': CSSSelectorPresenceChecker,
            'name': NamePresenceChecker,
            'xpath': XPathPresenceChecker
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
        return 'presence_checker'
