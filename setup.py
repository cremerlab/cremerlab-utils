from setuptools import setup, find_packages
import pathlib

__version__ = "0.0.3"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="cremerlab-utils",
    version=__version__,
    long_description=README,
    description="Python utilities for members of the Jonas Cremer lab at Stanford University",
    long_description_content_type='text/markdown',
    url="https://github.com/cremerlab/cremerlab-utils",
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    author="Griffin Chure",
    author_email="griffinchure@gmail.com",
    packages=find_packages(exclude=('docs', 'docsrc', 'exploratory', 'cremerlab.egg-info')),
    include_package_data=True,
    # package_data={"ecoli_gene_dict":["package_data/coli_gene_dict.pkl"]},
    # zip_safe=False,
)
