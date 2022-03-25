# 🎉🎉🎉 Winners 🎉🎉🎉

Authors:
* G. Farrow
* R. Laker
* P. Moloney
* T. Woolley

## AIHack 2022

Our GitHub Repo for the [AIHack2022](https://2022.aihack.org) hackathon.

## Submission

To see our final model and conclusions read `submission.ipynb` or our `Hackathon22` presentation.

## Project Description

* We attempted to optimise the cumulative profit from a battery that can charge and discharge (selling and buying energy at a time varying price).
* We trained the algorithm by rewarding it if the price it sold at was higher than the "expected price" for the current day, which we took to be the median price from the previous 24 hours. The sliding window median tracks the baseline price of energy over a 24 hour period while averaging over the daily fluctuations, without being biased by outliers of exceptional surge pricing.
* The model was trained on the first month of 2019 and then tested on the remaining dataset.
* We found that for the default hyper-parameters, that our trained model significantly outperformed a random walk in charge and discharge, obtaining a net profit of roughly £15000.
* The strategy adopted by the model was to buy low and sell high, which was especially profitable during the exceptional surge pricing.
* Further improvements to the project would have been:
    * To tune the model hyper-parameters by using a Bayesian optimisation algorithm. This is an appropriate tool as Bayesian optimisation can efficiently minimise a costly to evaluate function in many dimensions.
    * To improve the reward assignment so that the agent was more inclined to sell at peak prices, perhaps by including a gradient estimation of the current price.
    * To improve upon the forecasting method to estimate the price baseline for the next 24 hour period, rather than using the median of the previous 24hr period.

## Installation

Install the conda environment by running:

```bash
conda env create -f environment.yml
```

If you need to install a new package, try using conda first. Then use pip, but run it like this so that it installs only to the environment (not your base conda env)

```bash
python -m pip install <package>
```

I use pytest (installed in the environment) to run the tests. Either run `pytest` in the command line, or use the "Testing" extension of VSCode.

## Development

Only do these steps when contributing to the repo.

### Pre-commit

To make our code cleaner and well formatted I have included some [pre-commits](https://pre-commit.com/). The configuration for this is stored in `.pre-commit-config.yaml`. To install the hooks run:
```bash
pre-commit install
```

The hooks will then be installed when you make your first commit, or you can do:
```bash
pre-commit run
```

Try changing "badly_formatted.py" by adding some whitespace at the bottom then committing. It should make changes that you can then stage to commit.
