# FairPrep

FairPrep is a design and evaluation framework for fairness-enhancing interventions that treats data as a first-class citizen.

>  It empowers data scientists to conduct experiments on fairness-enhancing interventions with low effort, and at the same time enforces machine learning best practices.
>
> For more details, please refer to [FairPrep: Promoting Data to a First-Class Citizen in Studies on Fairness-Enhancing Interventions](https://dataresponsibly.github.io/documents/fairprep_short.pdf)

### Download and Initialize the environment for FairPrep

Step 1 Download FairPrep from [release-PyPI branch of FairPrep GitHub repository](https://github.com/DataResponsibly/FairPrep/tree/release-PyPI).

Step 2 Unzip the downloaded source file and initiate the python environment.

```bash
cd FairPrep-release-PyPI  # go to the unzipped repository that is just downloaded
python -m venv venv
source venv/bin/activate  # activate the environment for FairPrep
pip install -r requirements.txt
```

Step 3 Test the initialization of Fairprep
```bash
python -m unittest tests/test_FairPrep.py
```

#### Example in Jupyter Notebooks

There are some demos in `./notebooks/`. Initialize the jupyter notebook using the following command and go to the example notebook.

```bash
jupyter notebook
```

- [FairPrep__unit_pipeline.ipynb](notebooks/FairPrep__unit_pipeline.ipynb)
- TBC


### Assumptions for Your Own Dataset

1. The input dataset is a table that is stored in .CSV file.
2. We have some real datasets to demo the usage of FairPrep:
 - [German credit](https://github.com/Trusted-AI/AIF360/blob/master/aif360/data/raw/german/README.md)
 - [Adult income](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/adult)
 - [COMAS](https://github.com/Trusted-AI/AIF360/tree/master/aif360/data/raw/compas)

