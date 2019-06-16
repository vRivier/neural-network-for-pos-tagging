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


# data
dict_vectorizer = pickle.load(open('d_v','rb'))
label_encoder = pickle.load(open('l_e', 'rb'))

X_train = pickle.load(open('xdata', 'rb'))
y_train = pickle.load(open('ydata', 'rb'))

X_dev = pickle.load(open('xdev', 'rb'))
y_dev = pickle.load(open('ydev', 'rb'))


nn = pickle.load(open(args.nn_file,'rb'))


# tests
score = 0
for i in range(len(y_dev)):
	nn.feed_forward(X_dev[i].toarray()[0], y_dev[i])
	output = nn.layers[-1].a_layer.a_vector
	pred = list(output).index(max(output))
	if pred == y_dev[i]:
		score +=1 

print(score/len(y_dev))

print(nn.train_time/3600)
