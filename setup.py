#! /usr/bin/env python
"""
Set up for mymodule
"""
from setuptools import setup
import fnmatch
import os

def get_requirements():
    """
    Read the requirements from a file
    """
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt') as req:
            for line in req:
                # skip commented lines
                if not line.startswith('#'):
                    requirements.append(line.strip())
    return requirements

os.chdir("galess")
for root, dirnames, filenames in os.walk('data'):
	for filename in fnmatch.filter(filenames, '*'):
		package_info['package_data']['galess'].append(os.path.join(root, filename))
os.chdir("..")

setup(
    name='galess',
    version=0.1,
    description="Model for strong lens populations distribution estimates in surveys",
    author="Giovanni Ferrami",
    author_email="gferrami@student.unimelb.edu.au",
    url="https://github.com/Ferr013/GALESS",
    packages = ['galess', 'galess.LensStat', 'galess.Plots', 'galess.Utils', 'galess.ComputeSurveys'],
    install_requires=get_requirements(),
    python_requires='>=3.8',
    license="BSD-3"
)
