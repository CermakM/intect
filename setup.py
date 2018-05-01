import os
import sys

from setuptools import find_packages, setup

BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")

# When executing the setup.py, we need to be able to import ourselves, this
# means that we need to add the src/ directory to the sys.path.
sys.path.insert(0, SRC_DIR)

ABOUT = dict()
with open(os.path.join(SRC_DIR, 'poncoocr', '__about__.py')) as f:
    exec(f.read(), ABOUT)

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().split()

setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],

    author=ABOUT['__author__'],
    author_email=ABOUT['__email__'],
    url=ABOUT['__uri__'],

    license=ABOUT['__license__'],

    description=ABOUT['__summary__'],
    long_description="The architect allows to build, train and export servable neural network."
                     " Using single YAML file it is possible to fully define an architecture of"
                     " a neural network. The NN can then be trained, exported and served using"
                     " provided API.",

    classifiers=[
        "Development Status :: 1 - Planning"
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],

    package_data={
        'poncoocr': ['src/data/models', 'src/data/architectures']
    },
    include_package_data=True,

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    entry_points={
        'console_scripts': [
            'intect = poncoocr.api.cli:main',
            'intect-client = poncoocr.api.client:main'
        ],
    },

    install_requires=REQUIREMENTS,
)
