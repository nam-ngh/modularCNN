import numpy as np
from tqdm import tqdm
from typing import Literal

class ConvolutionalLayer:
    def __init__(self, no_of_filters=1, filter_size=3, stride=1, pad=0, input_shape=None):
        self.no_of_filters = no_of_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.filters = np.random.randn(no_of_filters, filter_size, filter_size, input_shape[-1]) * 0.01 # initialise the set of filters
        self.ftmap_size = int((input_shape[0] - self.filter_size + 2*pad)/self.stride) + 1 # determine the size of output feature map
        self.input = None

    def forwardprop(self, image):
        # pad the 4 edges of the image with the provided number of pixels
        if self.pad > 0:
            image = np.pad(image, [(self.pad,self.pad), (self.pad,self.pad), (0,0)], 'constant')
        
        # store image to later backprop:
        self.input = image

        # initialise output matrix with zeros:
        output = np.zeros(shape=(self.ftmap_size, self.ftmap_size, self.no_of_filters))
        
        # (x,y) = position of the sliding window on the input image and (i,j) = updated element in the output:
        y = j = 0
        # slide the window and scan to map the outputs:
        while (y + self.filter_size) <= image.shape[1]:
            x = i = 0
            while (x + self.filter_size) <= image.shape[0]:
                window = image[x:(x+self.filter_size), y:(y+self.filter_size), :]
                # apply each filter to the defined window
                for f in range(self.no_of_filters):
                    output[i,j,f] = np.sum(self.filters[f]*window)
                x += self.stride
                i += 1
            y += self.stride
            j += 1
        return output
    
    def backprop(self, dL_dout, learn_rate):
        # initialise input gradients:
        dL_din = np.zeros(shape=(self.input.shape))
        # initialise filter gradients:
        dL_dw = np.zeros(shape=self.filters.shape)

        #convolve over input to compute gradients:
        y = j = 0
        while (y + self.filter_size) <= self.input.shape[1]:
            x = i = 0
            while (x + self.filter_size) <= self.input.shape[0]:
                window = self.input[x:(x+self.filter_size), y:(y+self.filter_size), :]
                # apply each filter to the defined window
                for f in range(self.no_of_filters):
                    dL_dw[f] += dL_dout[i,j,f] * window # each stride adds a little to the filters gradient
                    dL_din[x:(x+self.filter_size), y:(y+self.filter_size),:] +=  dL_dout[i,j,f] * self.filters[f]
                x += self.stride
                i += 1
            y += self.stride
            j += 1
        
        #update filters:
        self.filters -= dL_dw * learn_rate
        return dL_din

class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
    
    def forwardprop(self, sample):
        self.input = sample 
        ftmap_size = int((sample.shape[0] - self.pool_size)/self.stride) + 1
        output = np.zeros(shape=(ftmap_size, ftmap_size, sample.shape[-1]))

        # slide the window and scan to map the outputs:
        y = j = 0
        while (y + self.pool_size) <= sample.shape[1]:
            x = i = 0
            while (x + self.pool_size) <= sample.shape[0]:
                # pool each channel:
                for c in range(sample.shape[-1]):
                    window = sample[x:(x+self.pool_size), y:(y+self.pool_size), c]
                    output[i,j,c] = np.max(window)
                x += self.stride
                i += 1
            y += self.stride
            j += 1
        return output
    
    def backprop(self, dL_dout, learn_rate):
        dL_din = np.zeros(shape=self.input.shape) # initialise derivatives matrix

        y = j = 0
        while (y + self.pool_size) <= dL_din.shape[1]:
            x = i = 0
            while (x + self.pool_size) <= dL_din.shape[0]:
                for c in range(dL_din.shape[-1]):
                    window = self.input[x:(x+self.pool_size), y:(y+self.pool_size), c]
                    x_idx, y_idx = np.where(window == np.max(window)) # get the index of max value inside input window
                    dL_din[(x+x_idx), (y+y_idx), c] = dL_dout[i,j,c]
                x += self.stride
                i += 1
            y += self.stride
            j += 1
        return dL_din

class ActivationLayer:
    fns = Literal['relu','leakyrelu','sigmoid','softmax']
    def __init__(self, fn:fns='relu', alpha = 0.0001):
        self.function = fn
        self.input = None
        self.output = None
        self.alpha = alpha # for leaky relu only

    def forwardprop(self, input_arr):
        self.input = input_arr

        if self.function == 'relu':
            output = np.maximum(input_arr, 0)
        
        if self.function == 'leakyrelu':
            output = np.maximum(input_arr, self.alpha*input_arr)

        if self.function == 'sigmoid':
            output = 1/(1+np.exp(input_arr))
        
        if self.function == 'softmax':
            output = np.exp(input_arr)/np.sum(np.exp(input_arr))

        self.output = output # store output to later derive
        return output
    
    def backprop(self, dL_dout, learn_rate):
        if self.function == 'relu':
            dout_din = np.int_(self.input > 0) # relu derivative
            dL_din = dL_dout * dout_din

        if self.function == 'leakyrelu':
            dout_din = np.divide(self.output, self.input, out=np.zeros_like(self.output), where=self.input!=0) # avoid divide by 0
            dL_din = dL_dout * dout_din

        if self.function == 'sigmoid':
            dout_din = self.output * (1-self.output) # sigmoid derivative
            dL_din = dL_dout * dout_din

        if self.function == 'softmax':
            output = self.output.reshape(-1,1)
            dout_din = np.diagflat(output) - np.dot(output, output.T) # derivative matrix of softmax function
            dL_din = np.dot(dout_din,dL_dout) # compute input loss deriv. by chain rule

        return dL_din
     
class FlattenLayer:
    def __init__(self):
        self.input_shape = None
    
    def forwardprop(self, input_arr):
        self.input_shape = input_arr.shape
        return input_arr.reshape(-1,1)
    
    def backprop(self, dL_dout, learn_rate):
        return dL_dout.reshape(self.input_shape)
        
        
class DenseLayer: 
    def __init__(self, units_in, units_out, init_weights_stdev=0.1):
        # we can choose what type of init method to use for the layer (kaiming, xavier, or just random) by passing in init_weights_stdev
        self.weights = np.random.randn(units_out, units_in) * init_weights_stdev
        self.biases = np.zeros(shape=(units_out,1))

    def forwardprop(self, input_arr):
        self.input = input_arr # store input for later use in backprop
        return np.dot(self.weights, input_arr) + self.biases
    
    def backprop(self, dL_dout, learn_rate):
        dL_dw = np.dot(dL_dout, self.input.T) # weight loss

        # update parameters:
        self.weights -= dL_dw * learn_rate
        self.biases -= dL_dout * learn_rate # bias loss gradient = dL_dout

        return np.dot(self.weights.T, dL_dout) # return derivative of loss wrt input, i.e. dL_din
    
class NN:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forwardpass(self, image):
        for layer in self.layers:
            image = layer.forwardprop(image)
        return image

    def backpass(self, dL_dout, learn_rate):
        for layer in reversed(self.layers):
            dL_dout = layer.backprop(dL_dout, learn_rate)

    def train(self, x_train, y_train, epochs, learn_rate):
        y_size = np.max(y_train)+1 # vector size of one-hot encoded y
        
        for epoch in range(epochs):
            loss_sum = 0
            correct_pred = 0
            for i in tqdm(range(y_train.shape[0]), ncols = 80):
                x = x_train[i]
                y = y_train[i]
                # one hot encode y: 
                y_1hot = np.zeros(shape=(y_size,1))
                y_1hot[y] = 1
                
                # pass the image through the network to obtain the probabilities array of the image belonging in each class:
                p = self.forwardpass(x) # (1x10) shape
                
                # keep track of correct predictions:
                if np.argmax(p) == y:
                    correct_pred += 1

                # compute cross-entropy loss:
                loss_sum += -np.log(p[y,0]) # -log of probability for the correct class
                gradient = np.divide(-y_1hot,p) #derivative of cross-entropy loss function - dL_dout, (1x10) shape

                # pass the gradient back through the network to adjust weights and biases:
                self.backpass(gradient, learn_rate)
                
            print(f'Epoch: {epoch}, Loss: {loss_sum/y_train.shape[0]}, Accuracy: {correct_pred*100/y_train.shape[0]}%')