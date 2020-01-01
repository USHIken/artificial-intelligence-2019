from utils import *
import pandas as pd
import matplotlib.pyplot as plt

class MultilayerPerceptron(object):

    def __init__(self, n_hidden=10, epochs=10, eta=1, shuffle=True, seed=None):
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.eta = eta

        # n_output = np.unique(y_train).shape[0]
        n_output = 1
        n_features = 1 # X_train.shape[1]

        # 入力層->隠れ層 への重み
        self.b_1 = np.zeros(self.n_hidden)
        self.w_1 = np.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # 隠れ層->出力層 への重み
        self.b_2 = np.zeros(n_output)
        self.w_2 = np.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

    def _forward(self, X):
        # 隠れ層の総入力
        z_in = np.dot(X, self.w_1) + self.b_1
        # 隠れ層の総出力
        z_out = np.tanh(z_in)
        # 出力層の総入力
        y_in = np.dot(z_out, self.w_2) + self.b_2
        # 出力層の総出力
        y_out = y_in

        return z_in, z_out, y_in, y_out
    
    def _onehot(self, y, n_output):
        onehot = np.zeros((n_output, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _tanh_derivate(self, a):

        return 1 - np.tanh(a)**2
    
    def fit(self, X_train, y_train):

        # 学習
        for _ in range(self.epochs):
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for idx in indices:

                # 順伝搬
                z_in, z_out, y_in, y_out = self._forward(X_train[idx])
                if y_out[0][0] == np.nan: exit(0)
                print("y_train, y_out = {}, {}".format(y_train[idx], y_out))

                # 逆伝搬
                sigma_1 = y_out - y_train[idx]
                delta_w_2 = np.dot(z_out.T, sigma_1)
                delta_b_2 = np.sum(sigma_1, axis=0)
                self.w_2 += self.eta * delta_w_2
                self.b_2 += self.eta * delta_b_2

                sigma_2 = np.dot(sigma_1, self.w_2.T) * self._tanh_derivate(z_in)
                delta_w_1 = np.dot(sigma_2, X_train[idx])
                delta_b_1 = np.sum(sigma_2, axis=0)
                self.w_1 += self.eta * delta_w_1
                self.b_1 += self.eta * delta_b_1

                print("w_1", self.w_1)
                print("w_2", self.w_2)
        return self
    
    def predict(self, X):
        _, _, _, y_out = self._forward(X)
        y_pred = np.argmax(y_out, axis=1)

        return y_pred

def util_mlp():
    N = 1000
    train = make_sin_data(N)
    test = make_sin_data(N)
    mlp = MultilayerPerceptron(epochs=1, eta=0.5)
    mlp.fit(train.index, train.values)
    valid_y = mlp.predict(np.array([test.index]).T)
    valid = pd.Series(valid_y, index=test.index)

    train_ax = plt.subplot(1,2,1)
    valid_ax = plt.subplot(1,2,2)
    # axes = [train_ax, valid_ax]
    plot_data(train, "train", axes=train_ax)
    plot_data(valid, "valid", axes=valid_ax)
    plt.show()

if __name__ == "__main__":
    # TODO: adaptive learning late
    util_mlp()
    pass
