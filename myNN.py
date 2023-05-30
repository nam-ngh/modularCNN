import numpy as np
from tqdm import tqdm
from typing import Literal

class ConvolutionalLayer:
    def __init__(self, no_of_filters=1, filter_size=3, stride=1, pad=0, input_shape=None):
        self.no_of_filters = no_of_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.input_shape = input_shape
        self.channels = input_shape[-1]
        # initialise the set of filters:
        self.filters = np.random.randn(filter_size, filter_size, input_shape[-1], no_of_filters) * 0.01
        self.filter_volume = filter_size*filter_size*input_shape[-1]
        # determine the size of the output feature map:
        self.output_size = int((input_shape[0] - filter_size + 2*pad)/stride) + 1
        # keeping track of each input
        self.input = None
        '''
        To follow along the code for this layer, we use an example of:
        - input_shape = (32,32,3)
        - no_of_filters = 16,
        - filter_size = 5,
        - stride = 3, pad = 3
        
        The filters have a shape of (5,5,3,16), i.e. 16 different filters of shape (5,5,3)
        Each filter_volume is therefore 5*5*3 = 75

        The output_size from this layer should be (32 - 5 + 2*3)/3 + 1 = 12
        Since there are 16 filters applied, output shape should therefore be (12,12,16)

        '''
    
    def forwardprop(self, image):
        # pad the 4 edges of the image with the provided number of pixels
        if self.pad > 0:
            image = np.pad(image, [(self.pad,self.pad), (self.pad,self.pad), (0,0)], 'constant')
        
        # store image to later backprop:
        self.input = image # self.input now has shape (38,38,3)

        # create a windowed view of the image, each window being the size of the filter (5,5,3), there are 12*12 windows in total. 
        # Each window and filter contribute to one output unit:
        windowed_image = np.lib.stride_tricks.as_strided(image,
                                                         shape=(self.output_size,self.output_size,
                                                                self.filter_size,self.filter_size,
                                                                self.channels),
                                                         strides=(image.strides[0]*self.stride,
                                                                  image.strides[1]*self.stride,
                                                                  image.strides[0],
                                                                  image.strides[1],
                                                                  image.strides[2]))
        # windowed_image shape: (12,12,5,5,3)
        # reshape each window into a row and filters into column for vectorized operation:
        flat_windows = windowed_image.reshape(self.output_size,self.output_size,self.filter_volume).reshape(self.output_size**2,self.filter_volume)
        flat_filters = self.filters.reshape(self.filter_volume, self.no_of_filters)
        output = np.dot(flat_windows,flat_filters).reshape(self.output_size,self.output_size,self.no_of_filters)
        return output
    
    def backprop(self, dL_dout, learn_rate):
        '''
        Two main tasks in a convolutional layer back propagation:
        - Find the filters gradients and adjust the filters accordingly with the learning rate
        - Find the input gradients to feed back to the previous layers

        1. The filter gradient matrix dL_df can be determined by:
        - Spacing out the dL_dout matrix by the stride value: from shape (12,12,16) to (34,34,16) with 0s in between
        - Use this matrix as the filter and perform convolution on the input (38,38,3). This operation should output a matrix 
        of shape (5,5,3,16), which is the shape of the filters
        '''
        # initialise the expanded output gradient matrix:
        m, n, f = dL_dout.shape # i.e. m, n = self.ouput_size, f = self.no_of_filters
        expanded_dL_dout = np.zeros((m*self.stride - (self.stride - 1),
                                     n*self.stride - (self.stride - 1),
                                     f), dtype=dL_dout.dtype)
        
        # replace values at stride intervals with dL_dout to complete the expanded matrix:
        expanded_dL_dout[::self.stride,::self.stride,:] = dL_dout
        # let:
        x = expanded_dL_dout.shape[0]

        # we now need to generate a windowed view of the input. Each window should be the 2D size of expanded_dL_dout (x,x):
        windowed_input = np.lib.stride_tricks.as_strided(self.input,
                                                         shape=(self.filter_size, self.filter_size, x, x, self.channels),
                                                         strides=(self.input.strides[0],
                                                                  self.input.strides[1],
                                                                  self.input.strides[0],
                                                                  self.input.strides[1],
                                                                  self.input.strides[2]))
        # windowed_input shape: (5,5,34,34,3)
        # transpose to get shape (5,5,3,34,34) then flatten to 2D for matrix multiplication:
        flat_windows = windowed_input.transpose(0,1,4,2,3).reshape(self.filter_volume, x**2)
        flat_dL_dout = expanded_dL_dout.reshape(x**2,f)
        dL_df = np.dot(flat_windows, flat_dL_dout).reshape(self.filters.shape)

        '''
        2. The input gradient matrix can be determined by:
        - Spacing out dL_dout as previously AND inversing it on the m, n axis
        - Using this matrix as the filter to perform FULL convolution on self.filters,
        i.e. convolution on self.filters padded with enough 0s on the first 2 axes

        Using our example this would be: convolving (34,34,16) on (71,71,3,16) to output (38,38,3),
        which is the shape of self.input
        '''
        # inversing and flattening dL_dout:
        flipped_dL_dout = np.flip(expanded_dL_dout,(0,1)).reshape(-1)

        # next we pad the filters with (x-1) 0s on either side of the first 2 dimensions:
        padded_filters = np.pad(self.filters, [(x-1,x-1), (x-1,x-1), (0,0), (0,0)], 'constant')
        # generate a windowed view of padded_filters. Each window being of size x:
        windowed_filters = np.lib.stride_tricks.as_strided(padded_filters,
                                                         shape=(self.input.shape[0], self.input.shape[0],
                                                                x, x, self.channels, f),
                                                         strides=(padded_filters.strides[0],
                                                                  padded_filters.strides[1],
                                                                  padded_filters.strides[0],
                                                                  padded_filters.strides[1],
                                                                  padded_filters.strides[2],
                                                                  padded_filters.strides[3]))
        # windowed_filters shape: (38,38,34,34,3,16)
        # transpose to get shape (38,38,3,34,34,16) then flatten to 2D for matrix multiplication:
        flat_filters = windowed_filters.transpose(0,1,4,2,3,5).reshape((self.input.shape[0]**2)*self.channels,-1)
        dL_din = np.dot(flat_filters, flipped_dL_dout).reshape(self.input.shape)

        # update filters and return dL_din:
        self.filters -= dL_df * learn_rate
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
    fns: Literal['relu','leakyrelu','sigmoid','softmax']
    def __init__(self, fn: fns='relu', alpha=0.0001):
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
                x = np.array(x_train[i], order='C') # making sure data is stored in row major order for ConvolutionalLayer to work
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