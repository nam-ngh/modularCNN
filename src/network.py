# main import:
import numpy as np

# extra imports:
# 1. to measure training progress:
from tqdm import tqdm
# 2. to pretty-print tables:
from tabulate import tabulate

class Net:
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