import nnfs
import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(100,3)

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap="brg")
plt.show()

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


class Loss:

    def calculate(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss


class Loss_categoricalCrossentrophy(Loss):
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample),y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods


layer1 = LayerDense(2,5)
layer1.forward_prop(X)
relu = ReLU()
relu.forward(layer1.output)
print(relu.output)
softmax = Softmax()
softmax.forward(layer1.output)
print(softmax.output)
loss_function = Loss_categoricalCrossentrophy()
loss = loss_function.calculate(softmax.output, y)

