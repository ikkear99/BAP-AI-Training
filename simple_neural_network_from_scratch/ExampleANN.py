from simple_neural_network_from_scratch import Sequential
import numpy as np
import h5py
import scipy.io as sio
from sklearn.model_selection import train_test_split

def load_mnist():
    # Load data
    classes = 10
    data = sio.loadmat("MNIST.mat")
    X = data["X"]
    y = data["y"]
    y = y.reshape(len(y))
    y -= 1
    X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    return X_train, y_train, X_test, y_test, classes

def load_dataset():
    """Loads the Cat vs Non-Cat dataset

      Returns
      -------
      X_train, y_train, X_test, y_test, classes: Arrays
        Dataset split into train and test with classes
    """

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':

    X_train, y_train, X_test, y_test, classes = load_dataset()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = X_train / 255.
    X_test = X_test / 255.

    random_state = 42
    epochs = 1000

    model = []

    model.append(Sequential.Dense(input_dim=X_train.shape[1], units=400, activation='relu', random_state=random_state))
    model.append(Sequential.Dense(input_dim=400, units=200, activation='relu', random_state=random_state))
    model.append(Sequential.Dense(input_dim=200, units=100, activation='relu', random_state=random_state))
    model.append(Sequential.Dense(input_dim=100, units=20, activation='relu', random_state=random_state))
    model.append(Sequential.Dense(input_dim=20, units=y_train.shape[1], activation='sigmoid', random_state=random_state))

    model = Sequential.Dense.fit(self=None,model=model, X_train=X_train,
                                 y_train=y_train, X_test=X_test, y_test=y_test, epochs=epochs, learning_rate=5e-3)

