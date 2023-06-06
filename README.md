# Convolutional Neural Network (CNN) modules from scratch

### About modularCNN
*Project Description:*<br>
The main purpose of this project is to efficiently facilitate the construction and implementation of CNN architectures from scratch, without the use of existing deep learning libraries such as keras or pytorch. Using this library, you can endlessly experiment with building various neural network architectures for image classification, from a simple one-hidden-layer perceptron to a deep network of multiple Convolutional and Pooling layers, with the appropriate activation function and weights initialisation strategy - all while the code underneath is purely python and numpy.

*Project background:*<br>
After building and training many CNN architectures using Keras's Sequential class, I wanted to deep dive into how neural networks actually work under the hood, and it turned out to be just a clever combination of linear algebra with differentiation and the chain rule. Therefore, it was possible to build my own model with nothing but numpy and python. However, my initial attempts, similar to most other implementations from scratch, took advantage of heavily nested for loops to carry out convolution, which is how convolution works in theory, but is borderline useless in practice since it is just way too slow. After further research, I came across numpy's as_strided function and tried implementing this in place of all for loops in each layer, which also allowed for vectorised matrix multiplication operations. These techniques together improved performance by about 300x on just the Convolutional layer's forwardprop function alone. Then, the rest of the functions and layers were optimised with the same principles (even though back propagation was much more complicated to work with), and here we are, a functional and moderately efficient means to build and train CNNs from scratch!

### Instalation and Use Guide
To work with this library on your local machine/environment, simply:
1. Clone the repository,

```
!git clone https://github.com/nam-ngh/modularCNN.git
```

2. Import modules:


```
from modularCNN.lib import layer, network
```

3. Now you can easily build your own neural networks, for example:
Defining your model:

```
model = network.Net()
```

Adding in layers:

```
model.add(layer.Convolutional(input_shape=(32,32,3),filters=16,filter_size=3,stride=1,pad=1))
# example of adding a Convolutional layer with 16 3x3 filters, stride of 1 and padding of 1 pixel
```
Get a summary of the architecture:

```
model.summary()
```

And train the model:

```
model.train(x_train, y_train, epochs=30, learn_rate=0.00001, val_size=0.1)
# split 10% of training data for validation
```







