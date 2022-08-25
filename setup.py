"""A setuptools based setup module for asediag"""
# -*- coding: utf-8 -*-

from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = [
    'pandas',
    'scipy',
    'setuptools',
    'netCDF4',
    'dask[complete]',
    'xarray',
    'cartopy==0.19.0.post1',
    'matplotlib',
    'cmaps',
    'pretty_html_table',
    'pytest',
]

test_requirements = [
    'pytest',
]

setup(
    name='asediag',
    version='1.1.0',
    description="Aerosol SE-data diagnostics tool",
    long_description=readme,
    author="Taufiq Hassan",
    author_email='taufiq.hassan@pnnl.gov',
    url='https://github.com/TaufiqHassan/asediag',
    packages=find_packages(exclude=['docs', 'tests']),
    package_data = {
            'adiags':['*'],
            },
    entry_points={
        'console_scripts':[
            'asediag=asediag.aer_diag_cli:main',
            ],
        },
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    tests_require=test_requirements,
)
