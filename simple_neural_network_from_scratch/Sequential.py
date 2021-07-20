import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Sequential:

    @staticmethod
    def initialize_layer_weights(n_l_1, n_l, random_state=0):
        """
        Initializes random weights and bias for a layer l
        :param random_state: random seed
        :param n_l_1: Number neurons in previous layer (l-1)
        :param n_l: Number neurons in previous layer l
        :return: Contains the randomly initialized weights and bias arrays
        """
        np.random.seed(random_state)

        wl = np.random.randn(n_l_1, n_l) * np.sqrt(2 / n_l_1)
        bl = np.random.randn(1, n_l) * np.sqrt(2 / n_l_1)

        return {'wl': wl, 'bl': bl}

class ActivationFunction(Sequential):

    @staticmethod
    def relu(Z):
        """Applies relu function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying relu function
        """

        return np.maximum(Z, 0)

    @staticmethod
    def relu_prime(Z):
        """Applies differentiation of relu function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying diff of relu function
        """

        return (Z > 0).astype(Z.dtype)

    @staticmethod
    def sigmoid(Z):
        """Applies sigmoid function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying sigmoid function
        """

        return 1 / (1 + np.power(np.e, -Z))

    @staticmethod
    def sigmoid_prime(Z):
        """Applies differentiation of sigmoid function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying diff of sigmoid function
        """

        return Z * (1 - Z)

    @staticmethod
    def leaky_relu(Z, alpha=0.01):
        """Applies leaky relu function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value
          alpha: float
            Negative slope coefficient

          Returns
          -------
          A: same shape as input
            Value after applying leaky relu function
        """

        return np.where(Z > 0, Z, Z * alpha)

    @staticmethod
    def leaky_relu_prime(Z, alpha=0.01):
        """Applies differentiation of leaky relu function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value
          alpha: float
            Negative slope coefficient

          Returns
          -------
          A: same shape as input
            Value after applying diff of leaky relu function
        """

        dz = np.ones_like(Z)
        dz[Z < 0] = alpha
        return dz

    @staticmethod
    def tanh(Z):
        """Applies tanh function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying tanh function
        """

        return np.tanh(Z)

    @staticmethod
    def tanh_prime(Z):
        """Applies differentiation of tanh function to an array/value

          Arguments
          ---------
          Z: float/int/array_like
            Original Value

          Returns
          -------
          A: same shape as input
            Value after applying diff of tanh function
        """

        return 1 - (np.tanh(Z) ** 2)

    @staticmethod
    def get_activation_function(name):

        """Returns function corresponding to an activation name

          Arguments
          ---------
          name: string
            'relu', 'leaky_relu', 'tanh' or 'sigmoid' activation

          Returns
          -------
          Corresponding activation function
        """

        if name == 'relu':
            return ActivationFunction.relu
        elif name == 'sigmoid':
            return ActivationFunction.sigmoid
        elif name == 'leaky_relu':
            return ActivationFunction.leaky_relu
        elif name == 'tanh':
            return ActivationFunction.tanh
        else:
            raise ValueError('Only "relu", "leaky_relu", "tanh" and "sigmoid" supported')

    @staticmethod
    def get_derivative_activation_function(name):

        """Returns differentiation function corresponding to an activation name

        Arguments
        ---------
        name: string
          'relu', 'leaky_relu', 'tanh' or 'sigmoid' activation

        Returns
        -------
        Corresponding diff of activation function
        """

        if name == 'relu':
            return ActivationFunction.relu_prime
        elif name == 'sigmoid':
            return ActivationFunction.sigmoid_prime
        elif name == 'leaky_relu':
            return ActivationFunction.leaky_relu_prime
        elif name == 'tanh':
            return ActivationFunction.tanh_prime
        else:
            raise ValueError('Only "relu", "leaky_relu", "tanh" and "sigmoid" supported')

class Dense(Sequential):
    """
        Returns a dense layer with randomly initialized weights and bias
        :param input_dim: Number of neurons in previous layer
        :param units: Number of neurons in the layer
        :param activation: Activation function to use. 'relu', 'leaky_relu', 'tanh' or 'sigmoid'
        :param random_state:
        :return:    Dense layer
                    An instance of the Dense layer initialized with random params
    """

    def __init__(self, input_dim, units, activation, random_state=0):

        params = Sequential.initialize_layer_weights(input_dim, units, random_state)

        self.units = units
        self.W = params['wl']
        self.b = params['bl']
        self.activation = activation
        self.Z = None
        self.A = None
        self.dz = None
        self.da = None
        self.dw = None
        self.db = None

    @staticmethod
    def forward_prop(X, model):

        for i in range(len(model)):

            if i == 0:
                X_l_1 = X.copy()
            else:
                X_l_1 = model[i - 1].A

            model[i].Z = np.dot(X_l_1, model[i].W) + model[i].b
            model[i].A = ActivationFunction.get_activation_function(model[i].activation)(model[i].Z)

        return model

    @staticmethod
    def calculate_loss(y, model):

        """Calculate the entropy loss

          Arguments
          ---------
          y: array-like
            True labels
          model: list
            List containing the layers

          Returns
          -------
          loss: float
            Entropy loss
        """

        m = y.shape[0]
        A = model[-1].A

        return np.squeeze(-(1. / m) * np.sum(np.multiply(y, np.log(A)) + np.multiply(np.log(1 - A),  1 - y)))


    @staticmethod
    def backward_prop(X, y, model):
        """
        Performs forward propagation
        :param X: Data
        :param y: True label
        :param model: List containing the layer
        :return: List containing layers with calculated 'dw', 'db'
        """

        m = X.shape[0]

        for i in range(len(model) - 1, -1, -1):

            if i == len(model) - 1:
                model[i].dz = model[-1].A - y
                model[i].dw = 1. / m * np.dot(model[i - 1].A.T, model[i].dz)
                model[i].db = 1. / m * np.sum(model[i].dz, axis=0, keepdims=True)

                model[i - 1].da = np.dot(model[i].dz, model[i].W.T)

            else:

                model[i].dz = np.multiply(np.int64(model[i].A > 0), model[i].da) * ActivationFunction.get_derivative_activation_function(
                    model[i].activation)(model[i].Z)

                if i != 0:
                    model[i].dw = 1. / m * np.dot(model[i - 1].A.T, model[i].dz)
                else:
                    model[i].dw = 1. / m * np.dot(X.T, model[i].dz)
                model[i].db = 1. / m * np.sum(model[i].dz, axis=0, keepdims=True)
                if i != 0:
                    model[i - 1].da = np.dot(model[i].dz, model[i].W.T)

        return model


    @staticmethod
    def update_weights(model, learning_rate):
        """
        Updates weights of the layers
        :param model: list containing the layers
        :param learning_rate: Learning rate for the weights update
        :return: List containing layers
        """
        for i in range(len(model)):
            model[i].W -= learning_rate * model[i].dw
            model[i].b -= learning_rate * model[i].db

        return model


    @staticmethod
    def predict(X, y, model):

        """Using the learned parameters, predicts a class for each example in X

        Arguments
        ---------
        X: array_like
          Data
        y: array_like
          True Labels
        model: list
          List containing the layers

        Returns
        -------
        predictions: array_like
          Vector of predictions of our model
        """

        model1 = Dense.forward_prop(X, model.copy())
        predictions = np.where(model1[-1].A > 0.5, 1, 0)

        return predictions

    @staticmethod
    def accuracy_score(X_test, y_test, model):
        """
        Accuracy of your test set
        :param y_test: np.ndarray
            True labels of your testing set
        :param y_pred: np.ndarray
            Predicted labels of your testing set
        :return: float
            Accuracy score of your test set
        """
        model1 = Dense.forward_prop(X_test, model.copy())
        predictions = np.where(model1[-1].A > 0.5, 1, 0)

        return sum(np.equal(y_test, predictions))[0] / len(y_test)


    def fit(self, model, X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01):
        """

        :param model:
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param epochs:
        :param learning_rate:
        :return:
        """
        cost_history = []
        for i in range(epochs):

            model = Dense.forward_prop(X_train, model)
            loss = Dense.calculate_loss(y_train, model)
            model = Dense.backward_prop(X_train, y_train, model)
            model = Dense.update_weights(model, learning_rate)

            print('Epoch: {}\tLoss: {:.6f}\tTrain Accuracy: {:.3f}\tTest Accuracy: {:.3f}'
                  .format(i, loss, accuracy_score(y_train, Dense.predict(X_train, y_train, model)),
                          accuracy_score(y_test, Dense.predict(X_test, y_test, model))))


            cost_history.append(loss)

        # Predict training set
        y_pred = Dense.predict(X_train, y_train, model)
        accuracy = accuracy_score(y_train, y_pred)
        print("Accuracy Train = ", accuracy)

        # Predict test set
        y_pred = Dense.predict(X_test, y_test, model)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy Test = ", accuracy)

        PLOT = True
        if PLOT:
            plt.plot(cost_history, label='train loss')
            plt.legend(loc='best')
            plt.show()

