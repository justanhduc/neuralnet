# Neuralnet

A high level framework for general purpose neural networks written in Theano.

## Requirements

[Theano](http://deeplearning.net/software/theano/) 1.0.1

[Scipy](https://www.scipy.org/install.html) 

[Numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)

[Matplotlib](https://matplotlib.org/)


## Installation
```
pip install --upgrade neuralnet
```

The version in this repo tends to be newer since I am lazy to make a new version available on Pypi when the change is tiny. TO install the version in this repo execute

```
pip install git+git://github.com/justanhduc/neuralnet.git@master
```

## Usages
To create a new model, simply make a new model class and inherit from Model in model.py. Please check out my [DenseNet](https://github.com/justanhduc/densenet) implementation for more details.
