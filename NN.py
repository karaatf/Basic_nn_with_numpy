import numpy as np


class OneHotMaker:
    def makeOneHot (self,inputs):
        self.one_hot = np.zeros((inputs.size,inputs.max()+1))
        self.one_hot[np.arange(inputs.size),inputs] = 1
        return self.one_hot


class DenseLayer:
    def __init__(self, n_inputs, n_neuron):
        self.W = 0.10 * np.random.randn(n_neuron,n_inputs)
        self.b = np.zeros((n_neuron, 1))
        self.A = 1
        self.Z = 1

    def forward_prop(self, X, function):

        self.Z = np.dot(self.W, X) + self.b

        if function == "relu":
            self.A = np.maximum(0, self.Z)

        elif function == "softmax":
            exp_values = np.exp(self.Z - np.max(self.Z, axis=1, keepdims=True))
            self.A = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.Z, self.A


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
            correct_confidence = y_pred_clipped[range(sample), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)

        negative_log_likelihoods = -np.log(correct_confidence)
        return negative_log_likelihoods


def dReLU(Z):
    return Z > 0


def backward_prop(Z1,Z2,A1,A2,W1,W2,X,Y,m):

    dZ2 = A2 - Y
    dW2 = (1 / m) * (dZ2.dot(A1.T))
    db2 = (1 / m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def parameterChange(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def train(X_train, Y_train, alpha,epochs, m):
    loss_function = Loss_categoricalCrossentrophy()
    layer1 = DenseLayer(784, 10)
    layer2 = DenseLayer(10, 10)
    for i in range(epochs):
        Z1, A1 = layer1.forward_prop(X_train, "relu")
        Z2, A2 = layer2.forward_prop(A1, "softmax")
        dW1, db1, dW2, db2 = backward_prop(Z1, Z2, A1, A2, layer1.W, layer2.W, X_train, Y_train, m)
        layer1.W, layer1.b, layer2.W, layer2.b = parameterChange(layer1.W, layer1.b, layer2.W, layer2.b, dW1, db1, dW2, db2, alpha)
        print("epochs: ", i)
        loss = loss_function.calculate(layer2.A, Y_train)
        print(loss)
    return layer1.W, layer2.W, layer1.b, layer2.b
