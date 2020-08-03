#!/usr/bin/env python3
"""single neuron performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoide(X):
    """sigmoide function"""
    return 1.0 / (1.0 + np.exp(-X))


class Neuron():
    """class Neuron that
    defines a single neuron"""
    def __init__(self, nx):
        """constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter of W"""
        return self.__W

    @property
    def b(self):
        """getter of b"""
        return self.__b

    @property
    def A(self):
        """getter of A"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation of the neuron"""
        self.__A = sigmoide(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of
        the model using logistic
        regression"""
        m = Y.shape[1]
        cost = (-1 / m) * (np.matmul(np.log(A), Y.T) +
                           np.matmul(np.log(1.0000001 - A), 1 - Y.T))
        return np.squeeze(cost)

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]

        error = A - Y
        dw = np.matmul(X, error.T) / m

        self.__W -= alpha * dw.T
        self.__b -= alpha * error.sum() / m

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron by updating the
        private attributes __W, __b, and __A"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = np.zeros(iterations + 1, )
        con = 0
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if verbose is True and (con == step or i == 0):
                print('Cost after {} iterations: {}'.format(i, cost))
                con = 0

            self.gradient_descent(X, Y, self.__A, alpha=alpha)
            if graph:
                costs[i] = cost
            con += 1

        predic, cost = self.evaluate(X, Y)

        if graph:
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.plot(np.arange(0, iterations + 1), costs)
            plt.show()
            """ plt.savefig('graph.png') """

        return predic, cost
