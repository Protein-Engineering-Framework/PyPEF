from setuptools import setup, find_packages


with open("requirements.txt", "r", encoding="utf-8") as install_requirements:
    requirements = install_requirements.read()

setup(
    name='pypef',
    version='0.1.7',
    author='Niklas Siedhoff & Alexander-Maurice Illig',
    author_email='n.siedhoff@biotec.rwth-aachen.de',
    license='CC BY-NC-SA 4.0',
    description='A command-line interface (CLI) tool for performing data-driven protein engineering by building '
                'machine learning models from sequence variant-fitness data (e.g., provided as CSV data). '
                'Get a small API provided for encoding and model validation from modules in pypef.api.',
    long_description='For detailed description including a short Jupyter Notebook-based '
                     'tutorial please refer to the GitHub page.',
    long_description_content_type='text/markdown',
    url='https://github.com/Protein-Engineering-Framework/PyPEF',
    py_modules=['pypef'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[requirements],
    python_requires='>=3.7',
    keywords='Pythonic Protein Engineering Framework',
    classifiers=[
        # 'Operating System :: POSIX :: Linux',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    entry_points={
        'console_scripts': [
            'pypef=pypef.cli.run:run_pypef'
        ],
    }
)
