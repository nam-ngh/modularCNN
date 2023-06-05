# main import:
import numpy as np

# extra imports:
# to measure training progress:
from tqdm import tqdm
# to show available options for method argument:
from typing import Literal
# to pretty-print tables:
from tabulate import tabulate

# layers:
class ConvolutionalLayer:
    def __init__(self, input_shape=None, filters=1, filter_size=3, stride=1, pad=0, pad_value=0):
        # specific layer attributes:
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.pad_value = pad_value
        self.channels = input_shape[2]
        # initialise the set of filters and compute each filter's volume:
        self.f = np.random.randn(filter_size, filter_size, input_shape[2], filters) * 0.01
        self.filter_volume = filter_size*filter_size*input_shape[2]
        # compute the size of the output feature map:
        self.output_size = int((input_shape[0] - filter_size + 2*pad)/stride) + 1
        # input cache:
        self.input = None
        # general layer attributes:
        self.name = 'Convolutional'
        self.input_shape = input_shape
        self.output_shape = tuple((self.output_size, self.output_size, self.filters))
        self.trainable_params = np.product(self.f.shape)
        '''
        To follow along the code for this layer, we use an example of:
        - input_shape = (32,32,3)
        - filters = 16,
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
            image = np.pad(image, [(self.pad,self.pad), (self.pad,self.pad), (0,0)], 
                           'constant', constant_values=self.pad_value)
        
        # store image in cache to later backprop:
        self.input = image # self.input now has shape (38,38,3)

        # create a windowed view of the image (i.e. convolution operation), each window being the shape of the filter (5,5,3).
        # there are 12*12 windows in total, each window and filter contribute to one output unit:
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
        flat_windows = windowed_image.reshape(self.output_size,
                                              self.output_size,
                                              self.filter_volume).reshape(self.output_size**2,
                                                                          self.filter_volume)
        
        flat_filters = self.f.reshape(self.filter_volume, self.filters)
        output = np.dot(flat_windows,flat_filters).reshape(self.output_size,
                                                           self.output_size,
                                                           self.filters)
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
        expanded_dL_dout = np.zeros((self.output_size*self.stride - (self.stride - 1),
                                     self.output_size*self.stride - (self.stride - 1),
                                     self.filters), dtype=dL_dout.dtype)
        
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
        flat_dL_dout = expanded_dL_dout.reshape(x**2,self.filters)
        dL_df = np.dot(flat_windows, flat_dL_dout).reshape(self.f.shape)

        '''
        2. The input gradient matrix can be mapped by performing FULL convolution on expanded_dL_dout, 
        using a flipped self.f as filters
        '''
        # flipping the filter on the first 2 dimensions:
        flipped_filters = np.flip(self.f, (0,1))

        # pad expanded_dL_dout with enough 0s on the first 2 dimensions:
        padded_dL_dout = np.pad(expanded_dL_dout, [(self.filter_size-1,self.filter_size-1),
                                                   (self.filter_size-1,self.filter_size-1),
                                                   (0,0)], 'constant')
        # generate a windowed view, each window the size of the filters, with the total number of windows same as input size:
        windowed_dL_dout = np.lib.stride_tricks.as_strided(padded_dL_dout,
                                                         shape=(self.input.shape[0], self.input.shape[0],
                                                                self.filter_size, self.filter_size,
                                                                self.filters),
                                                         strides=(padded_dL_dout.strides[0],
                                                                  padded_dL_dout.strides[1],
                                                                  padded_dL_dout.strides[0],
                                                                  padded_dL_dout.strides[1],
                                                                  padded_dL_dout.strides[2]))
        
        # shape: (38,38,5,5,16). Since dL ONLY needs to be passed to the original UNPADDED image,
        # we slice out this matrix's edge to only include the middle region of shape (32,32,5,5,16),
        # then flatten to (1444,400):
        windowed_dL_dout = windowed_dL_dout[self.pad:-self.pad, 
                                            self.pad:-self.pad, 
                                            :,:,:]
        flat_dL_dout_windows = windowed_dL_dout.reshape(self.input_shape[0]**2, (self.filter_size**2)*self.filters)

        # flipped_filters shape (5,5,3,16), transposed to (5,5,16,3) and flattened to (400, 3): 
        flat_filters = flipped_filters.transpose(0,1,3,2).reshape((self.filter_size**2)*self.filters,self.channels)
        dL_din = np.dot(flat_dL_dout_windows,flat_filters).reshape(self.input_shape)

        # finally, we update the filters and return dL_din:
        self.f -= dL_df * learn_rate
        return dL_din

class MaxPoolingLayer:
    def __init__(self, input_shape=None, pool_size=2):
        # specific layer attributes:
        self.pool_size = pool_size
        self.stride = pool_size
        self.output_size = int((input_shape[0] - pool_size)/pool_size) + 1
        self.channels = input_shape[2]
        # cache:
        self.pools = None
        # general layer attributes:
        self.name = 'MaxPooling'
        self.input_shape = input_shape
        self.output_shape = tuple((self.output_size, self.output_size, self.channels))
        self.trainable_params = 0
        '''
        Following the example from the previous layer:
        - input_shape = (12,12,16) i.e. ConvolutionalLayer output shape
        - Each pooling region is a 2x2 window, stride = 2
        - output_size = (12-2)/2 + 1 = 6
        - channels = 16
        Output of this layer hence has shape (6,6,16)
        '''
    def forwardprop(self, sample):
        # forming a windowed view of the input:
        windowed_sample = np.lib.stride_tricks.as_strided(sample,
                                                          shape=(self.output_size, self.output_size,
                                                                 self.pool_size, self.pool_size,
                                                                 self.channels),
                                                          strides=(sample.strides[0]*self.stride,
                                                                   sample.strides[1]*self.stride,
                                                                   sample.strides[0],
                                                                   sample.strides[1],
                                                                   sample.strides[2]))
        # windowed_sample shape: (6,6,2,2,16), store in cache to later use in backprop:
        self.pools = windowed_sample
        # return the one max value from each 2x2 window (axes 2 and 3) --> shape (6,6,16):
        return np.amax(windowed_sample, axis=(2,3))
    
    def backprop(self, dL_dout, learn_rate):
        '''
        This layer's backprop function takes the loss gradient from the previous layer and assign it to the position
        of the max value in each window defined in forwardprop. Gradients of non-max values are 0.
        '''
        # transform the input windowed view into a 2D array with 6x6x16 rows and 2x2 columns with the following steps:
        # - flatten each window, so shape (6,6,2,2,16) become (6,6,4,16)
        # - tranpose the last 2 axes so shape become (6,6,16,4)
        # - flatten from (6,6,16,4) to (576,4)
        # we now have a matrix where each row is a pooling window:
        flat_windows = self.pools.reshape(self.output_size,
                                          self.output_size,
                                          self.pool_size**2,
                                          self.channels).transpose(0,1,3,2).reshape(-1,self.pool_size**2)
        
        # next, we initialise dL_din and broadcast dL_dout to the argmax of each row in flat_windows:
        flat_dL_din = np.zeros(shape=(flat_windows.shape))
        flat_dL_din[range(flat_windows.shape[0]),np.argmax(flat_windows, axis = 1)] = dL_dout.reshape(-1)

        # now we need to reverse transform flat_dL_din into self.pools.shape by reversing the 3 steps above:
        windowed_dL_din = flat_dL_din.reshape(self.output_size,
                                              self.output_size,
                                              self.channels,
                                              self.pool_size**2).transpose(0,1,3,2).reshape(self.pools.shape)
        
        # now windowed_dL_din needs to be transformed to the original sample shape. This is where it gets more complicated without 
        # for loops if the pooling windows overlap, since the overlaps have to be added together to make dL_din.
        # without this, we only need to transpose and reshape:
        return windowed_dL_din.transpose(0,2,1,3,4).reshape(self.input_shape)

class ActivationLayer:
    def __init__(self, function: Literal['relu','leakyrelu','sigmoid','softmax']='relu', alpha=0.001):
        # specific layer attributes:
        self.function = function
        self.alpha = alpha # for leaky relu only
        # cache:
        self.input = None
        self.output = None
        # general layer attributes
        self.name = 'Activation: '+str(function)
        self.input_shape = '_' #
        self.output_shape = self.input_shape
        self.trainable_params = 0
        '''
        For this layer, there's no need to keep track of input and output shape since they're not needed for
        both forward and back prop, and they're always equal to each other.
        '''
    def forwardprop(self, input_arr):
        # store input to later derive:
        self.input = input_arr

        if self.function == 'relu':
            output = np.maximum(input_arr, 0)
        
        if self.function == 'leakyrelu':
            output = np.maximum(input_arr, self.alpha*input_arr)

        if self.function == 'sigmoid':
            output = 1/(1+np.exp(input_arr))
        
        if self.function == 'softmax':
            output = np.exp(input_arr)/np.sum(np.exp(input_arr))
        
        # store output to later derive:
        self.output = output
        return output
    
    def backprop(self, dL_dout, learn_rate):
        if self.function == 'relu':
            dout_din = (self.input > 0).astype(self.input.dtype) # relu derivative
            dL_din = dL_dout * dout_din

        if self.function == 'leakyrelu':
            dout_din = (self.input > 0).astype(self.input.dtype)
            dout_din[dout_din == 0] = self.alpha # replace all 0s with alpha
            dL_din = dL_dout * dout_din

        if self.function == 'sigmoid':
            dout_din = self.output * (1-self.output) # sigmoid derivative
            dL_din = dL_dout * dout_din

        if self.function == 'softmax':
            output = self.output.reshape(-1,1)
            dout_din = np.diagflat(output) - np.dot(output, output.transpose()) # derivative matrix of softmax function
            dL_din = np.dot(dout_din,dL_dout) # compute input loss deriv. by chain rule
        return dL_din
     
class FlattenLayer:
    def __init__(self, input_shape):
        # general layer attributes:
        self.name = 'Flatten'
        self.input_shape = input_shape
        self.output_shape = tuple((np.product(input_shape),1))
        self.trainable_params = 0
    
    def forwardprop(self, input_arr):
        # flatten to rows:
        return input_arr.reshape(-1,1)
    
    def backprop(self, dL_dout, learn_rate):
        return dL_dout.reshape(self.input_shape)
        
class DenseLayer: 
    def __init__(self, units_in, units_out, initial_Wvar=0.01):
        # specific layer attributes:
        self.weights = np.random.randn(units_out, units_in) * np.sqrt(initial_Wvar)
        self.biases = np.zeros(shape=(units_out,1))
        # cache:
        self.input = None
        # general layer attributes:
        self.name = 'Dense'
        self.input_shape = tuple((units_in,1))
        self.output_shape = tuple((units_out,1))
        self.trainable_params = np.product(self.weights.shape) + units_out # number of biases = units_out
    '''
    Dense a.k.a fully-connected layer:
    - units_in and units_out are the number of input and output neurons for the layer, respectively
    - e.g.: with units_in = 100 and units_out = 10, weights will have shape (10,100) and biases (10,1)
    - initial_Wvar specifies the variance for the initial weights distribution. Initializing each layer's weights appropriately could 
    help the network learn more stably and converge faster. Well known initialisation strategies include He, Xavier or random, which 
    can be implemented by passing in the appropriate initial_Wvar
    '''
    def forwardprop(self, input_arr):
        # store input to later backprop:
        self.input = input_arr
        return np.dot(self.weights, input_arr) + self.biases
    
    def backprop(self, dL_dout, learn_rate):
        dL_dw = np.dot(dL_dout, self.input.transpose()) # weight loss
        dL_din = np.dot(self.weights.transpose(), dL_dout) # input loss

        # update parameters and return input loss gradients:
        self.weights -= dL_dw * learn_rate
        self.biases -= dL_dout * learn_rate # bias loss gradient = dL_dout
        return dL_din

# network:    
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
    
    def summary(self):
        table = [['LAYER NAME','INPUT SHAPE','OUTPUT SHAPE ','TRAINABLE PARAMS']]
        total_params = 0
        for layer in (self.layers):
            lst = []
            lst.append(layer.name)
            lst.append(layer.input_shape)
            lst.append(layer.output_shape)
            lst.append(layer.trainable_params)
            # update total params:
            total_params += layer.trainable_params
            # update table:
            table.append(lst)
        table.append(['TOTAL','','',total_params])
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    def train(self, x_train, y_train, epochs=50, learn_rate=0.0001, val_size=0.1):
        # checks:
        assert (x_train.shape[0] == y_train.shape[0]), \
            f"Missing or incorrectly formatted data - got {x_train.shape[0]} inputs and {y_train.shape[0]} labels"
        assert (val_size >= 0 and val_size <= 1), f"Expected val_size value between 0 and 1, got: {val_size}"
        
        # define train and validation split:
        data_size = int(y_train.shape[0])
        trn_size = int(data_size*(1-val_size))

        # next, we one-hot encode y_train:
        outcomes = np.max(y_train)+1 # number of image categories 
        onehot_ytr = np.zeros(shape=(data_size,outcomes)) # initialise output matrix
        onehot_ytr[range(data_size),y_train] = 1 # set the yth value of each row from 0 to 1

        for epoch in range(epochs):
            trn_loss_sum = 0
            trn_correct_preds = 0
            # train the network on train data:
            for i in tqdm(range(trn_size), ncols = 100):
                x = np.array(x_train[i], order='C') # making sure data is stored in row major order for convolution to work
                y = onehot_ytr[i].reshape(10,1)
                
                # pass the image through the network to obtain the probabilities array of the image belonging in each class:
                p = self.forwardpass(x) # shape (10,1)
                
                # keep track of correct predictions:
                if np.argmax(p) == np.argmax(y):
                    trn_correct_preds += 1

                # compute cross-entropy loss:
                trn_loss_sum += -np.log(p[y==1][0]) # -log of probability for the correct class
                gradient = np.divide(-y,p) #derivative of cross-entropy loss function: dL_dout, shape (10,1)

                # pass the gradient back through the network to adjust weights and biases:
                self.backpass(gradient, learn_rate)
            
            # assess validation performance:
            if val_size > 0:
                print('Validating ...\r',end='')
                val_correct_preds = 0
                val_loss_sum = 0
                for i in range(trn_size,data_size):
                    # same as training but no backpass, i.e. no learning
                    x = np.array(x_train[i], order='C')
                    y = onehot_ytr[i].reshape(10,1)
                    p = self.forwardpass(x)

                    if np.argmax(p) == np.argmax(y):
                        val_correct_preds += 1
                    val_loss_sum += -np.log(p[y==1][0])

                print(f'Epoch: {epoch}, '
                      f'train_loss: {round((trn_loss_sum/trn_size), 8)}, '
                      f'train_acc.: {round((trn_correct_preds*100/trn_size), 5)}%, '
                      f'val_loss: {round((val_loss_sum/(data_size-trn_size)), 8)}, '
                      f'val_acc.: {round((val_correct_preds*100/(data_size-trn_size)), 5)}%', end='\n')
                
            elif val_size == 0: 
                print(f'Epoch: {epoch}, Loss: {round((trn_loss_sum/trn_size), 8)}, Accuracy: {round((trn_correct_preds*100/trn_size), 5)}%')