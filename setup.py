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
    long_description="",  # TODO

    classifiers=[
        "Development Status :: 1 - Planning"
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],

    package_dir={"": "src"},
    packages=find_packages(where='src'),

    install_requires=REQUIREMENTS,
    tests_require=[
       # TODO
    ],
)
