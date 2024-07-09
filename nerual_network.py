#Gerekli kütüphaneleri yüklüyorzu

import numpy as np


class OneHotMaker:
    def makeOneHot(self,inputs):
        self.one_hot = np.zeros((inputs.size,inputs.max()+1))
        self.one_hot[np.arange(inputs.size),inputs] = 1
        self.output = self.one_hot.T

class DenseLayer:
    def __init__(self, n_inputs, n_neuron):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neuron)
        self.biases = np.zeros((1, n_neuron))

    def forward_prop(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def ReLU(self, inputs):
        self.ReLU_output = np.maximum(0, inputs)

    def Softmax(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.Softmax_output = exp_values/np.sum(exp_values, axis=1, keepdims=True)

    def backward_prop(self):
        pass