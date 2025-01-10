import logging

import country_converter as coco
import pandas as pd

coco_logger = coco.logging.getLogger()
coco_logger.setLevel(logging.CRITICAL)

custom_data = pd.DataFrame.from_dict(
    {
        "name_short": ["Bonaire", "Netherlands Antilles", "Serbia", "East Timor"],
        "name_official": [
            "Bonaire, Saint Eustatius and Saba",
            "Netherlands Antilles",
            "Serbia",
            "East Timor",
        ],
        "regex": ["bonaire", "antilles", "serbia", "east timor"],
        "ISO3": ["BES", "ANT", "SER", "TLS"],
        "ISO2": ["a", "b", "c", "d"],
        "continent": ["America", "America", "Europe", "Asia"],
    }
)

name_converter = coco.CountryConverter(additional_data=custom_data)
