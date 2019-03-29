import numpy as np
from main import identity

class NN:

    """
    
    - nb_layers indique le nombre de couches du NN, y compris la couche d'input
    
    - nb_neurons indique le nombre de neurones de chaque couche
    C'est une liste de taille nb_layers avec à chaque position le nombre de neurones de la couche
    correspondante (y compris la couche d'input)
    
    """

    def __init__(self, nb_layers, nb_neurons, activations):

        self.layers = [Layer(nb_neurons[x], weights, activations[x]) for x in range(nb_layers)]
        # initialisation des poids?


    def forward_pass(self, example):
        """

        :param example:
        :return:
        """

        self.layers[0].outputs = example

        # itération sur les couches
        for i in range(1, len(self.layers)):
            
            previous_layer = self.layers[i-1]
            current_layer = self.layers[i]
            
            current_layer.inputs = np.dot(previous_layer.outputs, current_layer.weights)
            current_layer.outputs = current_layer.activation(current_layer.inputs)

        return current_layer.outputs


class Layer:

    def __init__(self, nb_neurons, activation):

        self.nb_neurons = nb_neurons
        self.weights = None
        self.inputs = None
        self.activation = activation
        self.outputs = None


