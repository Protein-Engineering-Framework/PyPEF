from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r", encoding="utf-8") as fh_single_core:
    with open("requirements_parallelization.txt", "r", encoding="utf-8") as fh_multithreading:
        requirements = fh_single_core.read() + fh_multithreading.read()

setup(
    name='pypef',
    version='0.1.1',
    author='Niklas Siedhoff & Alexander-Maurice Illig',
    author_email='n.siedhoff@biotec.rwth-aachen.de',
    license='CC BY-NC-SA 4.0',
    description='A CLI tool (not intended for API use) for performing data-driven protein engineering by building '
                'machine learning models from sequence variant-fitness data (e.g., provided as CSV data).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/niklases/PyPEF',
    py_modules=['pypef'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[requirements],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    entry_points={
        'console_scripts': [
            'pypef=pypef.run_pypef:run'
        ],
    }
)
