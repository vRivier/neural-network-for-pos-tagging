from NN import NeuralNetwork
import numpy as np

import argparse
import pickle
import time


parser = argparse.ArgumentParser()
parser.add_argument('nn_file')
args = parser.parse_args()


def ReLu(z_vector):
	return np.maximum(0,z_vector)

def softmax(z_vector):
    e = np.exp(z_vector)
    return(e/np.sum(e))

def negative_log_likelihood(output, y):
    return -np.log(output[y])


# - - - - - - - - - - - - - - - -
# hyper-paramètres

nb_neurons = [137173, 5, 18]

nn = NeuralNetwork(nb_neurons=nb_neurons, activation_function=ReLu,
                   output_function=softmax, loss_function=negative_log_likelihood)


# data
X_train = pickle.load(open('xdata', 'rb'))
y_train = pickle.load(open('ydata', 'rb'))


# train
t = time.time()

nn.train(X_train, y_train, 2, 0.01, 48)

nn.train_time = time.time() - t
pickle.dump(nn, open(args.nn_file, 'wb'))

