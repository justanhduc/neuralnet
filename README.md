# Neuralnet

A high level framework for general purpose neural networks written in Theano.
In my experience, running Visdom on Linux causes sementation fault. Therefore, this version is made to work in Linux. 
Except the exclusion of Visdom, everything is exactly the same as the master branch.  

## Requirements

[Theano](http://deeplearning.net/software/theano/)

[Scipy](https://www.scipy.org/install.html) 

[Numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)

[Matplotlib](https://matplotlib.org/)

[tqdm](https://github.com/tqdm/tqdm)


## Installation
To install a stable version, use the following command

```
pip install neuralnet
```

The version in this repo tends to be newer since I am lazy to make a new version available on Pypi when the change is tiny. 
To install the version in this repo execute

```
pip install git+git://github.com/justanhduc/neuralnet.git@linux (--ignore-installed) (--no-deps)
```

## Pretrained Models
[VGG16](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/vgg16_from_pytorch.npz)

[VGG19](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/vgg19_from_pytorch.npz)

[ResNet18](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/resnet18_from_pytorch.npz)

[ResNet34](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/resnet34_from_pytorch.npz)

[ResNet50](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/resnet50_from_pytorch.npz)

[ResNet101](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/resnet101_from_pytorch.npz)

[ResNet152](https://s3.ap-northeast-2.amazonaws.com/pretrained-theano-models/resnet152_from_pytorch.npz)

All the tests for these models are available at [test.py](https://github.com/justanhduc/neuralnet/tree/master/neuralnet/test.py).
Also, checkout the file for example usages.
