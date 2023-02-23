![ExoMDN](banner.png "Rapid characterization of exoplanet interiors")
# Rapid characterization of exoplanet interiors with Mixture Density Networks
![MIT License](https://img.shields.io/github/license/philippbaumeister/MDN_exoplanets.svg?style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7670706-blue?style=flat-square)](https://doi.org/10.5281/zenodo.7670706)

ExoMDN is a machine-learning-based exoplanet interior inference model using Mixture Density Networks. The model is 
trained on more than 5.6 million synthetic planet interior structures. Given mass, radius, and equilibrium 
temperature, ExoMDN is capable of providing a full inference of the interior structure of low-mass exoplanets in 
under a second without the need for a dedicated interior model.

This repository contains the trained models shown Baumeister & Tosi 2023 (submitted to A&A), as well as Python 
notebooks to load the models and run interior predictions of exoplanets. Interactive widgets are included 
to simplify loading an MDN model and running a prediction. 
We also make available the training routines in `more_examples/model_training_demo.ipynb`.

## Installation

### Using conda (preferred)

Create a new conda environment named ***exomdn*** from the `environment.yml` file, which installs all the required 
packages:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate exomdn
```

Then install the *exomdn* package by running
```
pip install -e .
```

### Using pip

This project requires Python 3.7 or higher.

Install the required packages and dependencies using
```
pip install -r requirements.txt
```

Then install the *exomdn* package by running
```
pip install -e .
```

### Required packages

- python>=3.7
- tensorflow=2.11
- tensorflow-probability=0.15
- scikit-learn=1.1.1
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- joblib
- ipywidgets
- jupyter

## Getting started

To get started check out `introduction.ipynb`. More in-depth examples can be found in the *more_examples* directory and 
more will be added over time.

## Acknowledgements
We are using the MDN layer for Keras by https://github.com/cpmpercussion/keras-mdn-layer 
