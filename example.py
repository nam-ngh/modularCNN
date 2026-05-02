from lib import layer, network

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'data'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def plot(x, y, label_names, y_hat=None, save_path=None):
    fig, ax = plt.subplots(2,3)
    i_list = np.random.randint(low=0, high=10000, size = 6) #generate list of random indices
    i=0
    for row in ax:
        for col in row:
            col.axis('off')
            #get image:
            img = x[i_list[i]]
            col.imshow(img)
            #get label:
            label_no = int(y[i_list[i]])
            label = label_names[label_no]
    
            if type(y_hat) == np.ndarray: # print predictions if they have been made
                predicted_label_no = int(y_hat[i_list[i]])
                predicted_label = label_names[predicted_label_no]
                col.set_title(
                    f'Index: {i_list[i]}\nLabel: {label}\nPredicted: {predicted_label}',
                    loc='left', fontdict={'fontsize': 8}
                )
                fig.suptitle(f'Test Accuracy {get_acc(y, y_hat)}%. Example Predictions:')
            else: 
                col.set_title(
                    f'Index: {i_list[i]}\nLabel: {label}',
                    loc='left', fontdict={'fontsize': 8}
                )
                fig.suptitle('Example Images')
            
            #update to the next generated index
            i+=1
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def load_cifar10():
    # List of train data files:
    data_batches = [
        f'{DATA_PATH}/data_batch_1', f'{DATA_PATH}/data_batch_2',
        f'{DATA_PATH}/data_batch_3', f'{DATA_PATH}/data_batch_4',
        f'{DATA_PATH}/data_batch_5',
    ]
    # Unpickling every batch in the list:
    train_data_dicts = [unpickle(batch) for batch in data_batches]
    # Unpickling test data file:
    test_data_dict = unpickle(f'{DATA_PATH}/test_batch')
    # Unpickling label names file:
    label_names_dicts = unpickle(f'{DATA_PATH}/batches.meta')
    label_names = [str(label, 'utf-8') for label in label_names_dicts[b'label_names']]

    # Group batches and define train data:
    x_train = np.empty(shape=(0,3072),dtype=np.uint8)
    y_train = []
    for dict in train_data_dicts:
        x_train = np.append(x_train, dict[b'data'], axis = 0)
        y_train += dict[b'labels']
    y_train = np.asarray(y_train)

    # Define test data:
    x_test = np.array(test_data_dict[b'data'])
    y_test = np.array(test_data_dict[b'labels'])

    # Convert x to image format:
    x_train = x_train.reshape(50000,3,32,32).transpose(0,2,3,1)
    x_test = x_test.reshape(10000,3,32,32).transpose(0,2,3,1)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # PREPROCESSING
    x_train_scaled = x_train / 255
    x_test_scaled = x_test / 255

    # random parallel shuffle train data for random validation:
    p = np.random.permutation(x_train.shape[0])
    rand_x_tr = x_train_scaled[p]
    rand_y_tr = y_train[p]

    return rand_x_tr, x_test_scaled, rand_y_tr, y_test, label_names

# predict new inputs:
def predict(model, X):
    y_pred = np.array([])
    for i in range(X.shape[0]):
        p = model.forwardpass(X[i])
        y_pred = np.append(y_pred, np.argmax(p))
    return y_pred.astype('int64')

# check similarity percentage between two arrays, i.e. vectorized accuracy calculation:
def get_acc(y_test, y_pred):
    diff = y_test - y_pred
    correct_preds = diff[diff==0].shape[0]
    acc = correct_preds*100/y_test.shape[0]
    print(f'Test Accuracy: {acc}%')
    return acc

def main():
    # BUILD NETWORK ARCHITECTURE
    cnn = network.Net()

    # get the model structure overview (at the moment empty):
    cnn.summary()

    # start adding layers
    cnn.add(layer.Convolutional(input_shape=(32,32,3),filters=16,filter_size=3,stride=1,pad=1))
    cnn.add(layer.Activation('relu'))
    cnn.add(layer.MaxPooling(input_shape=(32,32,16)))

    cnn.add(layer.Convolutional(input_shape=(16,16,16),filters=16,filter_size=3,stride=1,pad=1))
    cnn.add(layer.Activation('relu'))
    cnn.add(layer.MaxPooling(input_shape=(16,16,16)))

    cnn.add(layer.Convolutional(input_shape=(8,8,16),filters=32,filter_size=3,stride=1,pad=1))
    cnn.add(layer.Activation('leakyrelu'))
    cnn.add(layer.MaxPooling(input_shape=(8,8,32)))

    cnn.add(layer.Flatten(input_shape=(4,4,32)))
    cnn.add(layer.Dense(units_in=512,units_out=10,initial_Wvar=2/512))
    cnn.add(layer.Activation('softmax'))

    # view built architecture summary
    cnn.summary()

    # LOAD CIFAR 10
    x_train, x_test, y_train, y_test, label_names = load_cifar10()

    # TRAIN
    cnn.train(x_train, y_train, epochs=10, learn_rate=0.001, val_size=0.2)

    # TEST & PLOT EXAMPLES
    y_pred = predict(cnn, x_test)
    # un-normalise x_test then plot
    plot(
        (x_test * 255).astype(int), y_test.astype(int), 
        label_names=label_names, y_hat=y_pred,
        save_path='test_results.png'
    )

if __name__ == '__main__':
    main()