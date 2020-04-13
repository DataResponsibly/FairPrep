from setuptools import setup, find_packages

setup(name='fairprep',
      version='0.0.1',
      description='FairPrep is a design and evaluation framework for fairness-enhancing interventions that treats data as a first-class citizen.',
      url='https://github.com/DataResponsibly/FairPrep',
      author='DataResponsibly',
      author_email='',
      license='Apache',
      packages=find_packages(exclude=['tests']),
      python_requires='>=3.6'
     )
