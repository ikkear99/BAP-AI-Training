import numpy as np

class MF(object):
    def __init__(self, Y, K, X=None, W=None, lambda_=0.1, alpha=0.2, epochs=100):
        """

        :param Y: utility matrix
        :param K: X columns latent feature and W rows
        :param X: the latent feature (iterm) matrix
        :param W: the users matrix
        :param lambda_: regularization param avoid situation overfit (default 0.1)
        :param alpha: learning rate
        :param epochs: number of training loop (default 100)
        """
        self.__Y = Y
        # normalized data, update later in normalized_Y function
        self.__Y = self.__Y.copy()
        self.__K = K

        self.__lambda_ = lambda_
        self.__alpha = alpha
        self.__epochs = epochs

        # number of users, items, and ratings
        self.__users_count = int(np.max(Y[:, 0])) + 1
        self.__items_count = int(np.max(Y[:, 1])) + 1
        self.__ratings_count = Y.shape[0]
        self.__mu = np.zeros(self.__users_count)

        # random value iterm feature and weight matrix
        self.__X = np.random.randn(self.__items_count, K)
        self.__W = np.random.randn(K, self.__users_count)

    def normalized(self):
        """
        this method is used to normalized ratings
        :return:
        """
        mu = np.zeros(self.__users_count)

        for i in range(self.__users_count):
            indices_user_i = np.where(self.__Y[:, 0] == i)[0].astype(np.int32)
            ratings = []
            for j in indices_user_i:
                if float(self.__Y[j, 2]) != float(0):
                    ratings.append(self.__Y[j, 2])
            if len(ratings):
                _mean = np.mean(ratings)
            else:
                _mean = 0
            mu[i] = _mean
            # normalized
            if _mean != 0:
                for j in indices_user_i:
                    if float(self.__Y[j, 2]) != float(0):
                        self.__Y[j, 2] -= mu[i]
        self.__mu = mu

    def cost_function(self):
        """
        this method is used to calculate the cost function
        :return: cost function J
        """
        J = 0
        for i in range(self.__ratings_count):
            user = int(self.__Y[i, 0])
            item = int(self.__Y[i, 1])
            rate = self.__Y[i, 2]
            J += (1 / (2 * self.__ratings_count)) * np.square(rate - self.__X[item, :].dot(self.__W[:, user]))
        # regularized
        J += (self.__lambda_ / 2) * (
                np.linalg.norm(self.__X, ord="fro") + np.linalg.norm(self.__W, ord="fro"))
        return J

    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id and get the corresponding rates
        :param user_id: id of target user
        :return: array of item ids and ratings
        """
        indices_user = np.where(self.__Y[:, 0] == user_id)[0].astype(np.int32)
        item_ids = self.__Y[indices_user, 1].astype(np.int32)
        ratings = self.__Y[indices_user, 2].astype(np.float32)
        return item_ids, ratings

    def get_users_rating_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding rates
        :param item_id: id of item that need to find users who rated it
        :return: array of user ids and ratings
        """
        indices_item = np.where(self.__Y[:, 1] == item_id)[0].astype(np.int32)
        user_ids = self.__Y[indices_item, 0].astype(np.int32)
        ratings = self.__Y[indices_item, 2].astype(np.float32)
        return user_ids, ratings

    def update_x(self):
        """
        update rows of X matrix
        :return:
        """
        for i in range(self.__items_count):
            user_ids, ratings = self.get_users_rating_item(i)
            Wi = self.__W[:, user_ids]
            self.__X[i, :] = self.__X[i, :] - self.__alpha * (
                    -(1 / self.__ratings_count) * ((ratings - np.dot(self.__X[i, :], Wi)).dot(Wi.T))
                    + (self.__lambda_ * self.__X[i, :])).reshape((-1, self.__K))

    def update_w(self):
        """
        update columns of W matrix
        :return:
        """
        for i in range(self.__users_count):
            item_ids, ratings = self.get_items_rated_by_user(i)
            Xi = self.__X[item_ids, :]
            self.__W[:, i] = self.__W[:, i] - self.__alpha * (
                    -(1 / self.__ratings_count) * Xi.T.dot(ratings - Xi.dot(self.__W[:, i]))
                    + self.__lambda_ * self.__W[:, i]).reshape((self.__K,))

    def matrix_factorization(self):
        """
        implementation of matrix factorization algo
        :return:
        """
        self.normalized()
        for i in range(self.__epochs):
            self.update_x()
            self.update_w()
            mse_train = self.mse_evaluate(self.__Y)
            print("epoch:", i + 1, "cost:", self.cost_function(), "mse:", mse_train)

    def predict(self, user_id, item_id):
        """
        this method is used to make prediction about rating for item item_id of user user_id
        :param user_id: id of user target
        :param item_id: id of item target
        :return: prediction
        """
        pred_result = self.__X[item_id, :].dot(self.__W[:, user_id]) + self.__mu[user_id]
        # truncate if results are out of range [0, 10]
        if pred_result < 0:
            return 0
        elif pred_result > 10:
            return 10
        return pred_result

    def mse_evaluate(self, testing_set):
        """
        this method is used to evaluate the accuracy of our model using MSE
        :param testing_set: our dataset for testing
        :return: MSE
        """
        number_of_test = testing_set.shape[0]
        square_error = 0
        for i in range(number_of_test):
            prediction = self.predict(testing_set[i, 0], testing_set[i, 1])
            square_error += np.square(prediction - testing_set[i, 2])
        mean_square_error = square_error / number_of_test
        return mean_square_error

    def recommend(self, user_id):
        """
        Determine all items should be recommended for user u
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        indices_of_user = np.where(self.__Y[:, 0] == user_id)[0]
        items_rated_by_user = self.__Y[indices_of_user, 1].tolist()
        recommended_items = []
        for i in range(self.__items_count):
            if i not in items_rated_by_user:
                rating = self.predict(user_id, i)
                if rating > 9:
                    recommended_items.append(i)
        return recommended_items

    def print_recommendation(self, user_id):
        """
        print all items which should be recommended for each user
        """
        print('Recommendation: ')
        recommended_items = self.recommend(user_id)
        print('Recommend item(s):', recommended_items, 'to user', user_id)