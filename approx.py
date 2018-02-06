import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from sklearn.model_selection import train_test_split
import random


def ANN(activation='sigmoid'):
    model = Sequential()
    model.add(Dense(32, input_shape=(1, ),activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(512, activation=activation))
    model.add(Dense(256, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model


def main():
    X = np.arange(-10, 10.1, 0.001)
    Y = []
    Z = []
    for i in X:
        Y.append(math.sin(i)-math.cos(i)**2+i/2 + random.randint(-10, 10)/(random.random()*1000))
        Z.append(math.sin(i)-math.cos(i)**2+i/2)
    X = X.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    model = ANN()
    model.fit(np.ravel(x_train), np.ravel(y_train), epochs=30, batch_size=50)
    y_p = model.predict(x_test)
    
    l = sorted(zip(x_test, y_p), key= lambda x: x[0])
    X_p, Y_p = list(zip(*l))
    
    plt.plot(X, Y, c='g')
    plt.plot(X, Z, c='b')
    plt.plot(X_p, Y_p, c='r')
    plt.show()


if __name__ == "__main__":
    main()
