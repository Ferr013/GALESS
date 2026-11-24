#! /usr/bin/env python
from setuptools import setup


with open('requirements.txt') as req:
    requirements = [line.strip() for line in req if line.strip() and not line.startswith('#')]

# Include survey inputs and reference tables in both source and wheel installs.
data_files = ['data/SIE_dPdmu/*.txt', 'data/SIE_dPdmu/*/*.dat', 'data/surveys_params/*.param',
    'data/LENS_SEARCHES/*/*.csv', 'data/LENS_SEARCHES/*/*.tsv', 'data/LENS_SEARCHES/*/*.txt',
    'data/HMF_tables/*.txt', 'data/Collett2015/*.csv']

setup(
    name='galess',
    version='0.2',
    description="Model for strong lens populations distribution estimates in surveys",
    author="Giovanni Ferrami",
    author_email="gferrami@student.unimelb.edu.au",
    url="https://github.com/Ferr013/GALESS",
    packages=['galess', 'galess.LensStat', 'galess.Plots', 'galess.Utils',
              'galess.ComputeSurveys', 'galess.RareLensStat', 'galess.HOD'],
    package_data={'galess': data_files},
    install_requires=requirements,
    extras_require={'hod': ['hmf==3.4.4']},
    python_requires='>=3.8',
    license="BSD-3"
)
