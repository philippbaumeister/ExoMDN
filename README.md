![ExoMDN](banner.png "Rapid characterization of exoplanet interiors")
# Rapid characterization of exoplanet interiors with Mixture Density Networks
![MIT License](https://img.shields.io/github/license/philippbaumeister/MDN_exoplanets.svg?style=flat-square)

## Installation

### Using conda (preferred)

Create a new conda environment named ***exomdn*** from the `environment.yml` file, which installs all the required 
packages:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda env activate exomdn
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
- scikit-learn=0.11
- numpy
- pandas
- scipy
- matplotlib
- seaborn
- joblib
- ipywidgets
- jupyter

## Getting started

Interactive widgets are included in ExoMDN to simplify loading an MDN model and running a prediction. 
To get started check out `introduction.ipynb` in the *notebooks* directory. There you'll also find more in-depth 
examples.

## Acknowledgements
We are using the MDN layer for Keras by https://github.com/cpmpercussion/keras-mdn-layer 
