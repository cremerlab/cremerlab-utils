from setuptools import setup, find_packages
from os import path

__version__ = "0.0.001"

setup(
    name="cremerlab",
    version=__version__,
    description="",
    license="BSD",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    author="Griffin Chure",
    author_email="griffinchure@gmail.com",
    # include_package_data=True,
    # package_data={"ecoli_gene_dict":["package_data/coli_gene_dict.pkl"]},
    # zip_safe=False,
)
