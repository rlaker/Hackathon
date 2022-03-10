## AIHack 2022

Our GitHub Repo for the [AIHack2022](https://2022.aihack.org) hackathon.

## Installation

Install the conda environment by running:

```bash
conda env create -f environment.yml
```

This should install [sci-kit learn](https://scikit-learn.org/stable/index.html). This should be fine to start with, if we need to do some proper neural network stuff then we should use [Docker](https://www.docker.com/) container, as suggested by [Tensorflow](https://www.tensorflow.org/install).

If you need to install a new package, try using conda first. Then use pip, but run it like this so that it installs only to the environment (not your base conda env)

```bash
python -m pip install <package>
```

Test your install by running the ```test_installation.py``` script. I use pytest (installed in the environment) to run the tests. Either run `pytest` in the command line, or use the "Testing" extension of VSCode.

## Pre-commit

To make our code cleaner and well formatted I have included some [pre-commits](https://pre-commit.com/). The configuration for this is stored in `.pre-commit-config.yaml`. To install the hooks run:
```bash
pre-commit install
```

The hooks will then be installed when you make your first commit, or you can do:
```bash
pre-commit run
```

Try changing "badly_formatted.py" by adding some whitespace at the bottom then committing. It should make changes that you can then stage to commit.
