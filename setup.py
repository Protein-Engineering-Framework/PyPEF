#!/usr/bin/env python3
# for installation run me with:      pip install .
# or for editable/develop mode with: pip install -e .


import os
from setuptools import setup, find_packages

from pypef import __version__


directory_readme = os.path.join(os.path.dirname(__file__), "README.md")
with open(directory_readme, 'r') as readme:
    long_description = readme.read()

long_description = long_description.replace(".github/imgs/", "https://github.com/niklases/PyPEF/raw/main/.github/imgs/")

with open("requirements.txt", "r", encoding="utf-8") as install_requirements:
    requirements = install_requirements.readlines()

cleaned_requirements = ""
for requirement in requirements:
    if not requirement.startswith(('#', '--')):
        cleaned_requirements += requirement

setup(
    name='pypef',
    version=__version__.split('-')[0],
    author='Niklas Siedhoff & Alexander-Maurice Illig',
    author_email='niklas.siedhoff@rwth-aachen.de',
    license='CC BY-NC-SA 4.0',
    description='A command-line interface (CLI) tool for performing data-driven protein engineering '
                'by building machine learning (ML)-trained regression models from sequence variant '
                'fitness data (in CSV format) based on different techniques for protein sequence encoding. '
                'Next to building pure ML models, \"hybrid modeling\" is also possible using a blended '
                'model optimized for predictive contributions of a statistical and an ML-based prediction.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/niklases/PyPEF',
    py_modules=['pypef'],
    packages=find_packages(include=['pypef', 'pypef.*']),
    package_data={'pypef': ['ml/AAindex/*', 'ml/AAindex/Refined_cluster_indices_r0.93_r0.97/*']},
    include_package_data=True,
    install_requires=[cleaned_requirements],
    python_requires='>= 3.10, < 3.13',
    keywords='Pythonic Protein Engineering Framework',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    entry_points={
        'console_scripts': [
            'pypef = pypef.main:run_main'
        ],
    }
)
