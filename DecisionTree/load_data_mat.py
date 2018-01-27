'''Python file to load, randomise, and split Data provided as a mat file'''

import numpy as np
import scipy.io as sio

def shuffle_data(x,sd=42):
    '''Randomises data
    Input: x | numpy matrix, n x F
    Output: x | shuffled
    '''
    np.random.seed(seed=sd)
    np.random.shuffle(x)
    return x

def split_data(data, train_test = 0.7):
    '''Splits data into train and test.
    Inputs: 
        data | numpy matrix, n x F
        train_test | float, proportion of data which should be for training
    Outputs:
        train_data | np matrix
        test_data | np matrix
    '''
    data = shuffle_data(data)
    n = int(data.shape[0]*train_test)
    train_data = data[0:n,:]
    test_data = data[n:,:]
    return train_data, test_data

def load(file_path,train_test_split=0.7):
    '''Container function. Import this into main file'''
    data = sio.loadmat(file_path)

    x = np.asarray(data['x'])
    y = np.asarray(data['y'])

    x_train, x_validation = split_data(x, train_test_split)
    y_train, y_validation = split_data(y, train_test_split)
    
    return x_train, x_validation, y_train, y_validation

if __name__ == '__main__':
    x_train, x_val, y_train, y_val = load()
    print('Data Loaded')
