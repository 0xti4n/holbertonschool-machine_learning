#!/usr/bin/env python3
"""deep neural network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
    Activation Functions
    softmax, sigmoid, tanh
"""


def tanh_f(Z):
    """tanh activation"""
    res = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return res


def softmax(Z):
    """softmax activation"""
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=0, keepdims=True)


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


def activation(A_prev, W, b, activation):
    """Activation function"""
    if activation == 'sig':
        Z = linear_formula(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == 'tanh':
        Z = linear_formula(A_prev, W, b)
        A = tanh_f(Z)
    return A


class DeepNeuralNetwork():
    """deep neural network performing
    binary classification"""
    def __init__(self, nx, layers, activation='sig'):
        """constructor"""
        if activation != 'sig' or activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
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
        self.__activation = activation

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

    @property
    def activation(self):
        """getter of activation"""
        return self.__activation

    def forward_prop(self, X):
        """makes forward propagation"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            w = "W" + str(i)
            b = "b" + str(i)
            a = "A" + str(i - 1)
            Z = np.matmul(self.__weights[w],
                          self.__cache[a]) + self.__weights[b]
            a_new = "A" + str(i)
            if i != self.__L:
                if self.__activation == 'sig':
                    self.__cache[a_new] = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    self.__cache[a_new] = tanh_f(Z)
            else:
                t = np.exp(Z)
                a_new = "A" + str(i)
                self.__cache[a_new] = t / t.sum(axis=0, keepdims=True)
        Act = "A" + str(self.__L)
        return (self.__cache[Act], self.__cache)

    def cost(self, Y, A):
        """Calculates the cost of the model
        using logistic regression"""
        m = Y.shape[1]
        L_sum = np.sum(Y * np.log(A))
        cost = -(1 / m) * L_sum
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural
        network’s predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        value = np.amax(A, axis=0)
        predict = np.where(A == value, 1, 0)
        return predict, cost

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
                if self.__activation == 'sig':
                    error = sigmoid_back(dZ0, cache['A' + str(i+1)])
                elif self.__activation == 'tanh':
                    error = tanh_f(cache['A' + str(i+1)])
                    error = dZ0 * error
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

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """trains the model"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        _, cache = self.forward_prop(X)
        cost_list = []
        iter_x = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)
            if verbose is True and (
                    i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
                cost_list.append(cost)
                iter_x.append(i)
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)
                _, cache = self.forward_prop(X)
        if graph is True:
            plt.plot(iter_x, cost_list)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return (A, cost)

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
