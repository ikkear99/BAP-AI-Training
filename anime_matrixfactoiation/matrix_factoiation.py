import numpy as np


class MF(object):
    def __int__(self, Y, K, lambda_, X_init=None, W_init=None,
                learning_rate=0.5, max_item=1000):
        """

        @param Y:
        @param K:
        @param lambda_:
        @param X_init:
        @param W_init:
        @param learning_rate:
        @param max_item:
        @return:
        """
        self.Y = Y
        self.K = K
        self.lambda_ = lambda_
        self.X_init = X_init
        self.W_init = W_init
        self.learning_rate = learning_rate
        self.max_iter = max_item
        self.n_users = int(np.max(Y[:, 0])) + 1
        self.n_iterm = int(np.max(Y[:,1])) + 1
        self.n_ratings = Y.shape[0]
        self.X = np.random.randn(self.n_iterm, K) if X_init is None else X_init
        self.W = np.random.randn(K, self.n_users) if W_init is None else W_init
        self.b = np.random.randn(self.n_iterm)
        self.d = np.random.randn(self.n_users)

    def loss(self):
        """

        @return:
        """
        L = 0

        for i in range(self.n_ratings):
            # user_id, item_id, rating
            n, m, rating = int(self.Y[i, 0]), int(self.Y[i, 1]), self.Y[i, 2]
            L += 0.5 * (self.X[m].dot(self.W[:, n]) + self.b[m] + self.d[n] - rating) ** 2
            L /= self.n_ratings
        # regularization, don’t ever forget this
        return L + 0.5 * self.lambda_ * (np.sum(self.X ** 2) + np.sum(self.W ** 2))

    def updateXb(self):
        """

        @return:
        """
        for m in range(self.n_iterm):

            # get all users who rated item m and get the corresponding ratings
            ids = np.where(self.Y[:, 1] == m)[0]  # row indices of items m
            user_ids, ratings = self.Y[ids, 0].astype(np.int32), self.Y[ids, 2]
            Wm, dm = self.W[:, user_ids], self.d[user_ids]

            for i in range(30):  # 30 iteration for each sub problem
                xm = self.X[m]
                error = xm.dot(Wm) + self.b[m] + dm - ratings
                grad_xm = error.dot(Wm.T) / self.n_ratings + self.lambda_ * xm
                grad_bm = np.sum(error) / self.n_ratings
                # gradient descent
                self.X[m] -= self.learning_rate * grad_xm.reshape(-1)
                self.b[m] -= self.learning_rate * grad_bm

    def updateWd(self):  # and d
        for n in range(self.n_users):
            # get all items rated by user n, and the corresponding ratings
            ids = np.where(self.Y[:, 0] == n)[0]  # row indices of items rated by user n
            item_ids, ratings = self.Y[ids, 1].astype(np.int32), self.Y[ids, 2]
            Xn, bn = self.X[item_ids], self.b[item_ids]
            for i in range(30):  # 30 iteration for each sub problem
                wn = self.W[:, n]
                error = Xn.dot(wn) + bn + self.d[n] - ratings
                grad_wn = Xn.T.dot(error) / self.n_ratings + self.lambda_ * wn
                grad_dn = np.sum(error) / self.n_ratings
                # gradient descent
                self.W[:, n] -= self.learning_rate * grad_wn.reshape(-1)
                self.d[n] -= self.learning_rate * grad_dn


    def fit(self):
        for it in range(self.max_iter):
            self.updateWd()
            self.updateXb()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y)
                print('iter = % d, loss = % .4f, RMSE train = % .4 f' % (it + 1, self.loss(), rmse_train))