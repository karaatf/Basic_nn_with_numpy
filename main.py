import numpy as np
#Burada kerası mnist data setini daha rahat kullanmak için kullanıyoruz
from keras.datasets import mnist
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train = np.array(X_train).T
Y_train = np.array(Y_train)
print(X_train.shape)
print("-----------------------------------")
print(Y_train.shape)

#softmax katmanında hesaplanması için verilen Y değerlerini one hot olarak dönüştürür
from nerual_network import OneHotMaker
one_hot_make = OneHotMaker()
one_hot_make.makeOneHot(Y_train)
Y_train = one_hot_make.output


