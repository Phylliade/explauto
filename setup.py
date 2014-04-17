#!/usr/bin/env python

import re

from setuptools import setup, find_packages


def version():
    with open('explauto/_version.py') as f:
        return re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read()).group(1)


setup(name='explauto',
      version=version(),
      packages=find_packages(),

      install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas'],

      extra_require={
          'diva': ['pymatlab'],
          'imle': [],
      },

      setup_requires=['setuptools_git >= 0.3', ],

      #   include_package_data=True,
      #   exclude_package_data={'': ['README', '.gitignore']},

      zip_safe=True,

      author='Moulin-Frier Clement, Pierre Rouanet',
      author_email='clement.moulinfrier@gmail.com',
      description='Python Library for Autonomous Exploration',
      url='https://github.com/flowersteam/explauto',
      license='GNU GENERAL PUBLIC LICENSE Version 3',
      )
