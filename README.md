# Tidepool Data Science Simulator

#### -- Project Status: Active
#### -- Project Disclaimer: This work is Pre-Production

## Project Objective
The purpose of this project is to enable simulation of patient
metabolism and the interactions of controllers such as Tidepool Loop.
The current phase is for supporting FDA risk analysis. The longer term goal
is to support many activities such as Tidepool
Loop performance analysis and evaluation of algorithms for 
settings optimization.

## Definition of Done
The current phase of this project will be done when the code base has been
 tested thoroughly and the current FDA
risk analysis and iCGM risk sensitivity analysis have been
implemented in this environment.

## Project Description
During refactoring of the FDA risk analysis it became clear that
 increased generality/abstraction in an object-oriented
 approach to that code would support multiple projects of interest to Tidepool and
 Tidepool Data Science. The current state of this project is a generic simulator
that facilitates answering questions around Tidepool Loop risk, Tidepool
Loop performance, data-driven user modeling, and many others. 

The refactored code on which this is based is in `/notebooks/TEMPLATE_Run_Risk_Scenario_in_pyloopkit_in_colab_v0_5.ipynb`

### Technologies
* Python
* [Anaconda](https://www.anaconda.com/) for our virtual environments
* Pandas for working with data (99% of the time)
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
1. Current FDA scenarios are being kept `/data/raw/fda_risk_scenarios` within this repo.
2. Current demo use cases are the `/src`.

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
* 

## Important TODOs

* Integrate iCGM sensor and analysis scenarios (size: medium)
* Testing and cleanup (size: medium to large)
* Decouple input scenario format from Loop (size: medium)
* Integrate visuals and metrics repos (size: small)
* Surface all parameters for configuration (size: small)
* Control randomness via config seed(s) (size: small)
* Speed up Pyloopkit, possible in this code base (size: medium to large)

## Tidepool Data Science Team, Risk Assessment Team
|Name (with github link)    |  [Tidepool Slack](https://tidepoolorg.slack.com/)   |  
|---------|-----------------|
|[Ed Nykaza](https://github.com/[ed-nykaza])| @ed        |
|[Jason Meno](https://github.com/[jameno]) |  @jason    |
|[Cameron Summers](https://github.com/[scaubrey]) |  @Cameron Summers    |
|[Anna Quinlan] (https://github.com/novalegra) |
|[Eden Grown-Haeberli] (https://github.com/edengh) |
|[Shawn Foster] (https://github.com/ihadanidea) | @shawn |

