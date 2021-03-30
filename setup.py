from setuptools import setup, find_packages
import pathlib

__version__ = "0.0.01"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

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
    packages=find_packages(exclude=('docs', 'docsrc', 'exploratory')),
    include_package_data=True,
    # package_data={"ecoli_gene_dict":["package_data/coli_gene_dict.pkl"]},
    # zip_safe=False,
)
