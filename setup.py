from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="asediag",
    version="1.2.0",
    author="Taufiq Hassan",
    author_email="taufiq.hassan@pnnl.gov",
    description="Aerosol Diagnostics on Model Native Grid",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eagles-project/asediag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "cartopy",
        "matplotlib",
        "numpy",
        "pandas",
        "setuptools",
        "netcdf4",
        "h5netcdf",
        "xarray",
        "six",
        "dask",
    ],
    test_suite='test',
    tests_require=[
        'pytest',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'asediag=src.aer_diag_cli:main',
        ],
    },
)
