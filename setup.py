#!/usr/bin/env python3

from pathlib import Path

from setuptools import setup

directory = Path(__file__).parent.absolute()
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mitools",
    version="0.0.0",
    description="My Tools! <3",
    author="Sebastián Montagna",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "mitools",
        "mitools.scraping",
        "mitools.pandas",
        "mitools.nlp",
        "mitools.files",
        "mitools.etl",
        "mitools.economic_complexity",
        "mitools.country_utils",
        "mitools.context",
        "mitools.images",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[],
    python_requires=">=3.8",
    extras_require={
        "linting": [
            "flake8",
            "pylint",
            "mypy",
            "pre-commit",
        ],
        "testing": [
            "pytest",
            "pytest-xdist",
        ],
    },
    include_package_data=True,
)
