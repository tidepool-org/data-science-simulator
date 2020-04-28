# [Tidepool Data Science Project Template]
(Instructions: Everything in [ ] should be changed.)
(Everthing in ( )'s is instructions)

# [Project Name]

#### -- Project Status: [Active, On-Hold, Completed]
#### -- Project Disclaimer: This work is [Exploratory, Pre-Production, For-Production]

## Project Objective
The purpose of this project is to [___]. 

## Definition of Done
This phase of the project will be done when [___].

## Project Description
(Add a short paragraph with some details, Why?, How?, Link to Jira and/or Confluence)
In order to learn [___], we did [___].

### Technologies (Update this list)
* Python (99% of the time)
* [Anaconda](https://www.anaconda.com/) for our virtual environments
* Pandas for working with data (99% of the time)
* Google Colab for sharing examples
* Plotly for visualization
* Pytest for testing
* Travis for continuous integration testing
* Black for code style
* Flake8 for linting
* [Sphinx](https://www.sphinx-doc.org/en/master/) for documentation
* Numpy docstring format 


## Getting Started with the Conda Virtual Environment
1. Install [Miniconda](https://conda.io/miniconda.html). CAUTION for python virtual env users: Anaconda will automatically update your .bash_profile
so that conda is launched automatically when you open a terminal. You can deactivate with the command `conda deactivate` 
or you can edit your bash_profile. 
1. If you are new to [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/)
check out their getting started docs. 
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. In a terminal, navigate to the directory where you cloned this repo. 
1. Run `conda update -n base -c defaults conda` to update to the latest version of conda
1. Run `conda env create -f conda-environment.yml`. This will download all of the package dependencies
and install them in a conda (python) virtual environment.
1. Run `conda env list` to get a list of conda environments and select the environment
that was created from the environmental.yml file (hint: environment name is at the top of the file)
1. Run `conda activate <conda-env-name>` or `source activate <conda-env-name>` to start the environment.
1. Run `deactivate` to stop the environment.

## Getting Started with this project
1. Raw Data is being kept [here](Repo folder containing raw data) within this repo.
(If using offline data mention that and how they may obtain the data from the froup)
1. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
1. etc...

## Contributing Guide
1. All are welcome to contribute to this project.
1. Naming convention for notebooks is 
`[short_description]-[initials]-[date_created]-[version]`,
e.g. `initial_data_exploration-jqp-2020-04-25-v-0-1-0.ipynb`.
A short `_` delimited description, the creator's initials, date of creation, and a version number,  
1. Naming convention for data files, figures, and tables is 
`[PHI (if applicable)]-[short_description]-[date created or downloaded]-[code_version]`,
e.g. `raw_project_data_from_mnist-2020-04-25-v-0-1-0.csv`,
or `project_data_figure-2020-04-25-v-0-1-0.png`.

NOTE: PHI data is never stored in github and the .gitignore file includes this requirement as well.

## Featured Notebooks/Analysis/Deliverables
* [Colab Notebook/Figures/Website](link)

## Tidepool Data Science Team
|Name (with github link)    |  [Tidepool Slack](https://tidepoolorg.slack.com/)   |  
|---------|-----------------|
|[Ed Nykaza](https://github.com/[ed-nykaza])| @ed        |
|[Jason Meno](https://github.com/[jameno]) |  @jason    |
|[Cameron Summers](https://github.com/[scaubrey]) |  @Cameron Summers    |

