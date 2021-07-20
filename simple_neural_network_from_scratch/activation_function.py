import numpy as np
"""
Activation functions in neural networks
"""
class ActivationFunction:
    """
    Activation function
    """
    @staticmethod
    def sigmoid(z):
        """
        Value after applying relu function
        :param z: Original Value
        :return: Value after applying sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """
        Applies derivative of sigmoid function to an array/value
        :param z: Original Value
        :return:
        """
        x = self.sigmoid(z)
        return x*(1 - x)

    @staticmethod
    def tanh(z):
        """
        Applies tanh function to an array/value
        :param z: Original Value
        :return: Value after applying tanh function
        """
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def tanh_derivative(self, z):
        """
        Applies derivative of tanh function to an array/value
        :param z: Original Value
        :return: Value after applying der of tanh function
        """
        return 1 - self.tanh(z)**2

    @staticmethod
    def relu(z):
        """
        Applies relu function to an array/value
        :param z: Original Value
        :return: Value after applying relu function
        """
        return np.maximum(z, 0)

    @staticmethod
    def relu_derivative(z):
        """
        Applies derivative of relu function to an array/value
        :param z: Original Value
        :return: Value after applying der of sigmoid function
        """
        if z > 0:
            return 1
        else:
            return 0

    @staticmethod
    def leaky_relu(z, alpha = 0.01):
        """
        Applies leaky relu function to an array/value
        :param z: Original Value
        :param alpha: Negative slope coefficient
        :return: value after applying leaky relu function
        """
        return np.where(z > 0, z, z*alpha)

    @staticmethod
    def leaky_relu_derivative(z, alpha = 0.01):
        """
        Applies differentiation of leaky relu function
        :param z: Original Value
        :param alpha: Negative slope coefficient
        :return: Value after applying der of leaky relu function
        """
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz