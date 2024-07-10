import numpy as np
from NN import OneHotMaker,DenseLayer, dReLU, backward_prop, parameterChange, train

#Burada kerası mnist data setini daha rahat kullanmak için kullanıyoruz
from keras.datasets import mnist
m=10000


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train = np.array(X_train[:m])
X_train = (X_train.reshape(784,m)/255)
Y_train = np.array(Y_train[:m])



onehot = OneHotMaker()
onehoty = onehot.makeOneHot(Y_train)




W1, W2, b1, b2=train(X_train,onehoty.T,0.001,1500,m)