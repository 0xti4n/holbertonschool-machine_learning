#!/usr/bin/env python3
"""deep neural network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid_back(dz, cache):
    """sigmoid BP"""
    dS = cache * (1 - cache)
    return dz * dS


def sigmoid(X):
    """sigmoid Activation"""
    return 1.0 / (1.0 + np.exp(-X))


def linear_formula(A, W, b):
    """Linear formula"""
    Z = np.matmul(W, A) + b
    return Z


def activation(A_prev, W, b):
    """Activation function"""
    Z = linear_formula(A_prev, W, b)
    A = sigmoid(Z)
    return A


class DeepNeuralNetwork():
    """deep neural network performing
    binary classification"""
    def __init__(self, nx, layers):
        """constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if type(layers) is not list:
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lidx in range(self.__L):
            if type(layers[lidx]) is not int or layers[lidx] < 1:
                raise TypeError('layers must be a list of positive integers')

            self.__weights['b' + str(lidx+1)] = np.zeros((layers[lidx], 1))

            if lidx == 0:
                sqr = np.sqrt(2 / nx)
                formula = np.random.randn(layers[lidx], nx) * sqr
                self.__weights['W' + str(lidx+1)] = formula
            else:
                sqr = np.sqrt(2 / layers[lidx - 1])
                formula = np.random.randn(layers[lidx], layers[lidx - 1]) * sqr
                self.__weights['W' + str(lidx+1)] = formula

    @property
    def L(self):
        """getter of L"""
        return self.__L

    @property
    def cache(self):
        """getter of cache"""
        return self.__cache

    @property
    def weights(self):
        """getter of weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation
        of the neural network"""
        self.__cache['A0'] = X
        A = X
        for lidx in range(1, self.__L + 1):
            A_prev = A

            W = self.__weights['W' + str(lidx)]
            b = self.__weights['b' + str(lidx)]

            A = activation(A_prev, W, b)
            self.__cache['A' + str(lidx)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        m = Y.shape[1]
        cost = (-1 / m) * (np.matmul(np.log(A), Y.T) +
                           np.matmul(np.log(1.0000001 - A), 1 - Y.T))
        return np.squeeze(cost)

    def evaluate(self, X, Y):
        """Evaluates the neural
        network’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient
        descent on the neural network"""
        m = Y.shape[1]
        copy_weights = self.__weights.copy()
        layers = self.__L

        for i in reversed(range(layers)):
            if i == layers - 1:
                error = cache['A' + str(i+1)] - Y
                dw = np.matmul(cache['A' + str(i)], error.T) / m

            else:
                dZ0 = np.matmul(copy_weights['W' + str(i+2)].T, error)
                error = sigmoid_back(dZ0, cache['A' + str(i+1)])
                dw = np.matmul(error, cache['A' + str(i)].T) / m

            db = np.sum(error, axis=1, keepdims=True) / m

            if i == layers - 1:
                result = copy_weights['W' + str(i+1)] - alpha * dw.T
                self.__weights['W' + str(i+1)] = result

            else:
                result = copy_weights['W' + str(i+1)] - alpha * dw
                self.__weights['W' + str(i+1)] = result

            result = copy_weights['b' + str(i+1)] - alpha * db
            self.__weights['b' + str(i+1)] = result

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 and step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)
            costs.append(cost)
            self.gradient_descent(Y, self.__cache, alpha=alpha)
            if verbose is True:
                if i % step == 0:
                    print('Cost after {} iterations: {}'.format(i, cost))

        if graph:
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.plot(np.arange(0, iterations + 1), costs)
            """ plt.savefig('23-graphDNN.png') """
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object
        to a file in pickle format"""

        if '.pkl' not in filename:
            filename = filename + '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled
        DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            return None
