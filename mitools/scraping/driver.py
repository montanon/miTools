import os

from seleniumrequests import Chrome
from selenium import webdriver
from selenium.webdriver.support.ui import Select

from .driver_finders import *
from .driver_waiters import *
from .driver_checkers import *


CHROME_DRIVER = '~/WebBrowser_Drivers/chromedriver'
DOWNLOADS_DIRECTORY = os.path.expanduser('~/Downloads')



class DriverHandler:

    def __init__(self, use_options=False, headless=False, options_prefs=None):

        self.prefs = {
                "download.default_directory": DOWNLOADS_DIRECTORY,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.plugins_disabled": "Chrome PDF Viewer",
                "plugins.always_open_pdf_externally": True
            }
        if options_prefs:
            self.prefs.update(options_prefs)

        self.driver_options = self.configure_driver_options(use_options)

        self.headless = headless
        if self.headless:
            self.driver_options.add_argument("--headless")

        self.driver = self.init_driver()
        self.link = None

        self.init_attributes(FinderFactory())
        self.init_attributes(WaiterFactory())
        self.init_attributes(PresenceCheckerFactory())

    def init_driver(self):
        driver = Chrome(executable_path=CHROME_DRIVER, options=self.driver_options)
        if self.headless:
            driver.set_window_size(1920, 1080)
        return driver

    def configure_driver_options(self, use_options=False):
        options = webdriver.ChromeOptions()
        if use_options:
            options.add_experimental_option("prefs", self.prefs)

        return options

    def init_attributes(self, factory):
        creators = factory.creators
        factory_name = factory.name
        for creator in creators:
            attribute = self._set_creator_attribute(creator, factory, factory_name)
            self._set_creator_methods(creator, attribute)

    def _set_creator_attribute(self, creator, factory, factory_name):
        attribute_name = f"{creator}_{factory_name}"
        setattr(self, attribute_name, factory.get_object(creator))
        return getattr(self, attribute_name)

    def _set_creator_methods(self, creator, attribute):
        for method_name, method in self._get_callable_methods(attribute):
            setattr(self, f"{method_name}_{creator}", self.method_wrapper(method))

    def _get_callable_methods(self, attribute):
        for method_name in dir(attribute):
            if not method_name.startswith('__'):
                method = getattr(attribute, method_name)
                if callable(method):
                    yield method_name, method

    def method_wrapper(self, method):
        def wrapper(*args, method=method, **kwargs):
            return method(self.driver, *args, **kwargs)
        return wrapper

    def try_load_link_and_wait(self, link, id_to_wait_for=None):
        try:
            self.load_link_and_wait(link, id_to_wait_for)
        except TimeoutException as e:
            raise TimeoutException(e)
    
    def load_link_and_wait(self, link, id_to_wait_for=None):
            self.link = link
            self.driver.get(link)
            if id_to_wait_for:
                self.wait_for_element_id(id_to_wait_for, wait_time=10)

    def switch_to_n_tab(self, n):
        window_handles = self.driver.window_handles
        self.driver.switch_to.window(window_handles[n])

    def close_current_tab_and_go_home(self):
        if len(self.driver.window_handles) == 1:
            return
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[0])

    @staticmethod
    def click(element):
        element.click()

    @staticmethod
    def robust_click(element):
        pass

    @staticmethod
    def input_to_element(element, text):
        element.send_keys(text)

    @staticmethod
    def select_element_index(element, index):
        selector = Select(element)
        selector.select_by_index(index)

    @staticmethod
    def select_element_value(element, value):
        selector = Select(element)
        selector.select_by_value(value)

    @staticmethod
    def select_element_text(element, text):
        selector = Select(element)
        selector.select_by_visible_text(text)

    @staticmethod
    def get_element_selected_options(element):
        selector = Select(element)
        return selector.all_selected_options
        
    @staticmethod
    def get_all_element_options(element):
        selector = Select(element)
        return selector.options
    
    @staticmethod
    def deselect_element_options(element):
        selector = Select(element)
        selector.deselect_all()


def main():

    dh = DriverHandler(headless=True)
    print('='*40)
    print('Driver Handler attributes:')
    print('='*40)
    for att in dir(dh):
        print(att)
    print('-'*40)
    return dh


if __name__ == '__main__':
    dh = main()
