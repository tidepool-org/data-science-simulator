# Tidepool Data Science Simulator

#### -- Project Status: Active
#### -- Project Disclaimer: This work is in Production

## Project Objective
The purpose of this project is to enable simulation of patient
metabolism and the interactions of controllers such as Tidepool Loop.
The current phase is for supporting FDA risk analysis. The longer term goal
is to support many activities such as Tidepool
Loop performance analysis and evaluation of algorithms for 
settings optimization.

## Ongoing development
This project supports ongoing risk assessment for Tidepool Loop. Changes (including exploring proposed changes) to the
algorithm and new features all may require new development.

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
* JSLint for linting
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
1. Current FDA scenarios are being kept `scenario_configs/tidepool_risk_v2/loop_risk_v2_0` within this repo.
2. Exploratory iCGM sensitivity analyses are located in `tidepool_data_science_simulator/projects/icgm` within this repo.
3. Analysess of proposed Tidepool Loop therapy settings guardrails are located in `tidepool_data_science_simulator/projects/loop_guardrails` within this repo.

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

## Important TODOs

* Integrate iCGM sensor and analysis scenarios (size: medium)
* Testing and cleanup (size: medium to large)
* Integrate visuals and metrics repos (size: small)
* Surface all parameters for configuration (size: small)
* Control randomness via config seed(s) (size: small)
* Integrate NoisySensor selection in JSON config (size: small)
* Model Coastal pump delivery and support pump selection via JSON config (size: medium to large)
* Model autobolusing at varying percentages of recommended bolus (size: medium)

## Tidepool Contributors

## Current Tidepool Contributors
|Name (with github link)    |
|---------|
|[Shawn Foster] (https://github.com/[ihadanidea]) |
|[Pete Schwamb] (https://github.com/[ps2])  |

## Previous Tidepool Contributors
|Name (with github link)     |
|----------|
|[Ed Nykaza](https://github.com/[ed-nykaza])|
|[Jason Meno](https://github.com/[jameno]) |
|[Cameron Summers](https://github.com/[scaubrey]) |
|[Anna Quinlan] (https://github.com/[novalegra]) |
|[Eden Grown-Haeberli] (https://github.com/[edengh])  |

