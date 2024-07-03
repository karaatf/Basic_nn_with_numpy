import nnfs
import numpy as np
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(100,2)


class LayerDense:

    def __init__(self, n_inputs, n_neuron):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neuron)
        self.biases = np.zeros((1, n_neuron))

    def forward_prop(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:

    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)


layer1 = LayerDense(2,5)
layer1.forward_prop(X)
relu = ReLU()
relu.forward(layer1.output)
print(relu.output)
softmax = Softmax()
softmax.forward(layer1.output)
print(softmax.output)