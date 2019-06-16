#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""

    Neural Network class with implemented Feed Forward

    En terme de notation, au sein du réseau de neurone:
    - w réfère aux poids / paramètres connectant une couche à la précédente
    - z réfère aux valeurs des neurones avant activation
    - a réfère aux valeurs des neurones après activation

    Les couches sont constituées de deux sous-couches Z_layer et A_layer (respectivement avant et après activation, conformément aux notations)

    On distingue 3 types de couches différentes:
    - couche d'input, qui se définit uniquement par ses valeurs de sortie ou a_vector. Cette valeur est passée depuis l'extérieur du réseau.
    - couche cachée, connectées avec les poids (weights), avec des valeurs et une fonction d'activation pour la passe avant, et des valeurs pour la passe arrière
    - couche d'output, qui est une couche cachée avec une fonction d'activation propre

"""

import numpy as np
import random
import matplotlib.pyplot as plt


"""
learning rate
[0,1] au lieu de [-1,1]
sum() du biais
nb de layers / neurones
fonctions utilisées
"""


class NeuralNetwork:


    def __init__(self, nb_neurons, activation_function, output_function, loss_function):

        self.layers = list()

        # input layer
        self.layers.append(self.Layer(0, 0, None))
        # hidden layers
        self.layers.extend([self.Layer(nb_neurons[x], nb_neurons[x-1], activation_function=activation_function)
                            for x in range(1, len(nb_neurons)-1)])
        # output layer
        self.layers.append(self.Layer(nb_neurons[-1], nb_neurons[-2], output_function))

        self.loss_function = loss_function

        self.train_time = 0
        self.learning_rate = 0
        self.mini_batch = 0



    def feed_forward(self, input_vector, y=0):
        self.layers[0].a_layer.a_vector = input_vector

        for i in range(1, len(self.layers)):

            previous_layer = self.layers[i-1]
            layer = self.layers[i]

            # calcul z_vector
            input_vector = previous_layer.a_layer.a_vector
            weights = layer.z_layer.weights
            biases = layer.z_layer.biases
            layer.z_layer.z_vector = np.dot(previous_layer.a_layer.a_vector, weights) +biases

            # calcul a_vector
            activation_function = layer.a_layer.activation_function
            layer.a_layer.a_vector = activation_function(layer.z_layer.z_vector)


        output = self.layers[-1].a_layer.a_vector

        return output[y]



    # source : https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    # je reprends les notations pour les noms des variables :
    # - dE_da signifie dérivée partielle de l'Erreur par rapport à l'activation
    # - da_dz signifie dérivée partielle de l'activation par rapport à la préactivation
    # etc
    def output_derivation(self, y):
        output_layer = self.layers[-1]
        output = output_layer.a_layer.a_vector

        dE_da = soft_nll_derivative(output, y)
        da_dz = 1
        dz_dw = self.layers[-2].a_layer.a_vector

        output_layer.dE_dz += dE_da * da_dz
        output_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(output_layer.dE_dz))



    def hidden_derivation(self, i):
        hidden_layer = self.layers[i]
        a_vector = hidden_layer.a_layer.a_vector

        dE_da = np.dot(self.layers[i+1].dE_dz, self.layers[i+1].z_layer.weights.T)
        da_dz = ReLu_derivative(a_vector)
        dz_dw = self.layers[i-1].a_layer.a_vector

        hidden_layer.dE_dz += dE_da * da_dz
        hidden_layer.update_matrix += np.dot(np.asmatrix(dz_dw).T, np.asmatrix(hidden_layer.dE_dz))


    def backpropagation(self, y):

        # calculer les dérivées à chaque étapes
        self.output_derivation(y)

        for i in range(2, len(self.layers)):
            # on parcourt les layers dans l'ordre inverse
            self.hidden_derivation(-i)


    def update(self, learning_rate, mini_batch):

        for layer in self.layers[1:]:
            layer.z_layer.weights -= (layer.update_matrix /mini_batch) *learning_rate
            layer.z_layer.biases -= (np.sum(layer.dE_dz /mini_batch)) *learning_rate
            layer.reset_update()


    def train(self, X_train, y_train, epoch=1, learning_rate=0.5, mini_batch=1):
      
        self.learning_rate = learning_rate
        self.mini_batch = mini_batch

        for e in range(epoch):
            # zipd = list(zip(X_train, y_train))
            # random.shuffle(zipd)
            # X_train, y_train = zip(*zipd)

            for i in range(len(y_train)):
                self.feed_forward(X_train[i].toarray()[0])
                self.backpropagation(y_train[i])

                if i==1000:
                    print(1000)

                if i%mini_batch == 0:
                    self.update(learning_rate, mini_batch)


    def reset(self):
        for l in self.layers[1:]:
            l.z_layer.weights = np.random.rand(l.z_layer.weights.shape)
            l.z_layer.biases = np.random.rand(l.z_layer.biases.shape)


    # - - - - - - - - - - - - - - - - - - - - - - - -
    # classe pour représenter une couche (cachée ou de sortie)
    # contient les méthodes pour le calcul des vecteurs de pré-activation et d'activation
    class Layer:

        def __init__(self, nb_neurons, prev_nb_neurons, activation_function):
            self.prev_nb_neurons = prev_nb_neurons
            self.nb_neurons = nb_neurons
            self.z_layer = self.Z_layer(nb_neurons, prev_nb_neurons)
            self.a_layer = self.A_layer(activation_function)
            self.update_matrix = 0
            self.dE_dz = 0

        def reset_update(self):
            self.update_matrix = 0
            self.dE_dz = 0

        class Z_layer:
            def __init__(self, nb_neurons, prev_nb_neurons):
                self.weights = np.random.rand(prev_nb_neurons, nb_neurons)
                self.biases = np.random.rand(nb_neurons, )
                self.z_vector = 0

        class A_layer:
            def __init__(self, activation_function):
                self.activation_function = activation_function
                self.a_vector = 0





# dérivées des fonctions

# activation
def sigmoid_derivative(output):
    return output * (1 - output)

def tanh_derivative(dE_da):
    return 1 - dE_da**2

def ReLu_derivative(dE_da):
    return 1*(dE_da>0)

# perte
def mean_squared_derivative(output, y):
    return output - y

# output - perte
def soft_nll_derivative(output, y):
    e = np.zeros(len(output))
    e[y] += 1
    return output - e
