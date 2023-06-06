# Convolutional Neural Network (CNN) modules from scratch

## About modularCNN

The main purpose of this project is to efficiently facilitate the construction and implementation of CNN architectures from scratch, without the use of existing deep learning libraries such as keras or pytorch. Using this library, you can endlessly experiment with building various neural network architectures for image classification, from a simple one-hidden-layer perceptron to a deep network of multiple Convolutional and Pooling layers, with the appropriate activation function and weights initialisation strategy - all while the code underneath is purely python and numpy.

**Background:**<br>
After building and training many CNN architectures using Keras's Sequential class, I wanted to deep dive into how neural networks actually work under the hood, and found that it was possible to build my own model with nothing but numpy and python. However, my initial attempts, similar to most other implementations from scratch, took advantage of heavily nested for loops to carry out convolution, which is how convolution works in theory, but is borderline useless in practice since it was way too slow. After further research, I came across numpy's as_strided function and tried implementing this in place of all the for loops in each layer, which also allowed for vectorised matrix multiplication operations. These techniques together improved performance by about 300x on just the Convolutional layer's forwardprop function alone. Then, the rest of the functions and layers were optimised with the same principles (even though back propagation was much more complicated to work with), and here we are, a functional and moderately efficient means to build and train CNNs from scratch!

**Current status:**<br>
This project is under active development. Expected features include:
- Dropout layer
- Batch normalization
- Learn rate scheduling

## Instalation and Use Guide
To work with this library in your local notebook, simply:
1. Clone the repository,

```
!git clone https://github.com/nam-ngh/modularCNN.git
```

2. Import modules:

```
from modularCNN.lib import layer, network
```

3. Now you can easily build your own neural networks, for example:<br>
 - Defining your model with the Net class:

```
model = network.Net()
```

 - adding in layers:

```
model.add(layer.Convolutional(input_shape=(32,32,3),filters=16,filter_size=3,stride=1,pad=1))
# e.g.: a Convolutional layer with 16 3x3 filters, stride of 1 and padding of 1 pixel
```

 - get a summary of the architecture:

```
model.summary()
```

 - and train the model:

```
model.train(x_train, y_train, epochs=30, learn_rate=0.00001, val_size=0.1)
# split 10% of training data for validation
```

## Important Notes
- The current model and layers are only compatible with square images: Input feature sets x_train, x_test must be provided in shape (n,x,x,c) where n = number of sample, x = height and width of the image, and c = number of channels (c=1 for b&w, c=3 for RGB)
- input_shape must be specified for all layers upon adding to the network, in the form of (x,x,c), with the exception of Activation layer where this is not needed and Dense layer, where the number of neurons in and neurons out are required instead.
- Ouput size of Convolutional and MaxPooling layer can be determined as follows: o = (x - filter_size + 2*pad)/stride + 1. Please make sure this is a whole number so that convolutions are complete and free of errors
- It is recommmended that you determine the output shape of the previous layer first before adding the next, to make sure that the shapes don't mismatch. If you are unsure you can run the model summary function each time you add a layer to check the layers added so far
- Currently for MaxPooling layer, pool_size and stride can only be set as the same number under pool_size, i.e. pooling regions can't overlap






