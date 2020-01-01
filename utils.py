import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_linear_data(n, a=1, b=0, noise={"loc":0.0, "scale":0.1}):
    """
    線形データを返す

    Parameters
    ----------
    n : int
        データ数
    a : int
        線形関数の係数
    b : int
        線形関数の切片
    noise : dict, or None
        ノイズ関数のlocation(平均)とscale(分散)
    
    Returns
    -------
    data : pandas.core.series.Series, shape = [2, n]
        線形データ
    """
    func = lambda x: a*x + b
    data = make_data(n, func, -10, 10, noise)
    return data


def make_sin_data(n, noise={"loc":0.0, "scale":0.1}):
    """
    sinデータを返す

    Parameters
    ----------
    n : int
        データ数
    noise : dict, or None
        ノイズ関数のlocation(平均)とscale(分散)
    
    Returns
    -------
    data : pandas.core.series.Series, shape = [2, n]
        sinデータ
    """
    func = np.sin
    data = make_data(n, func, -1*np.pi, 1*np.pi, noise)
    return data


def make_data(n, func, start, end, noise):
    """
    関数の出力データを返す

    Parameters
    ----------
    n : int
        データ数
    func : function
        出力したいデータを処理する関数
    start : int
        データのx軸の始点
    end : int
        データのx軸の終点
    noise : dict, or None
        ノイズ関数のlocation(平均)とscale(分散)
    
    Returns
    -------
    data : pandas.core.series.Series, shape = [2, n]
        出力データ
    """
    x = np.linspace(start, end, n)
    data = pd.Series(func(x), index=x)
    if noise:
        noise_func = lambda data, n: data + np.random.normal(noise["loc"], noise["scale"], n)
        data = noise_func(data, n)
    return data


def linear_regression(data):
    """
    線形回帰を行った結果を返す

    Parameters
    ----------
    data : pandas.core.series.Series, shape = [2, n]
        ノイズを含む(x,y)のデータn個
    
    Returns
    -------
    data : pandas.core.series.Series, shape = [2, n]
        受け取ったデータから計算したパラメータで回帰を行った結果のデータ
    """

    N = data.shape[0]
    x = data.index
    y = data.values

    sigma_x = sum(x)
    sigma_y = sum(y)
    sigma_x_y = sum(x*y)
    sigma_x_x = sum(x*x)

    a = ((-N * sigma_x_y) + sigma_x_y) / (sigma_x**2 - (N * sigma_x_x))
    b = ((sigma_x_x * sigma_y) - (sigma_x * sigma_x_y)) / ((N * sigma_x_x) - sigma_x**2)

    data = make_linear_data(N, a, b, noise=False)
    return data


def polynomial_basis_func(x, param):
    """
    非線形回帰の基底関数(多項式)の出力結果を返す

    Parameters
    ----------
    x : pandas.core.series.Series, shape = [1, n]
        関数への入力データ
    param : int
        関数のパラメータ.ここでは返すデータの次元数
    
    Returns
    -------
    data : pandas.core.series.Series, shape = [n, param]
        基底関数の出力結果
    """

    N = x.shape[0]

    phi_x = np.zeros((N, param))
    for j in range(param):
        phi_x[:,j] = pow(x, j)
    
    return phi_x
    

def radial_basis_func(x, param, nu=10):
    """
    非線形回帰の基底関数(RBF)の出力結果を返す

    Parameters
    ----------
    x : pandas.core.series.Series, shape = [1, n]
        関数への入力データ
    param : int
        関数のパラメータ.ここでは返すデータの次元数
    nu : int, default 10
        RBFのパラメータ

    Returns
    -------
    data : pandas.core.series.Series, shape = [n, param]
        基底関数の出力結果
    """

    N = x.shape[0]
    A = np.array_split(x, param)

    c = [0]*param
    for j in range(param):
        c[j] = sum(A[j]) / len(A[j])

    s2 = [0]*param
    for j in range(param):
        s2[j] = np.linalg.norm(A[j] - c[j], ord=2)**2 / len(A[j])

    phi_x = np.ones((N, param+1))
    for j in range(param):
        phi_x[:,j+1] = np.exp(-((x-c[j])**2) / (nu * s2[j]))
    
    return phi_x


def nonlenear_regression(data, basis_func, param, regularizer_lambda=0, plot=None):
    """
    非線形回帰を行った結果を返す
    """


    if basis_func == polynomial_basis_func:
        param += 1  # j = 1 .. param

    N = data.shape[0]
    x = data.index
    y = data.values

    phi_x = basis_func(x, param)
    phi_x_t = phi_x.T
    regularizer = regularizer_lambda*np.eye(phi_x.shape[1]) # calc λI for regularization
    # W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x) + regularizer), phi_x_t), y)
    W = np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x) + regularizer), np.dot(phi_x_t, y))
    Y = np.zeros(N)
    for j in range(param):
        Y += W[j] * phi_x[:, j]

    # plotting curve
    if plot:
        for j in range(param):
            df = pd.DataFrame(pd.Series(W[j]*phi_x[:, j], index=x))
            df.plot(color='k', legend=None, ax=plot, style="--")

    data = pd.Series(Y, index=data.index)

    return data


class MultilayerPerceptron(object):

    def __init__(self, n_hidden=10, epochs=10, eta=1):
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

            for idx in indices:

                # 順伝搬
                z_in, z_out, y_in, y_out = self._forward(X_train[idx])
                print("y_train, y_out = {}, {}".format(y_train[idx], y_out))

                # 逆伝搬
                sigma_out = y_out - y_train[idx]
                delta_w_2 = np.dot(z_out.T, sigma_out)
                self.w_2 -= self.eta * delta_w_2
                delta_w_1 = np.dot(sigma_out, self.w_2.T) * np.dot(self._tanh_derivate(z_in), X_train[idx])
                self.w_1 -= self.eta * delta_w_1

        return self
    
    def predict(self, X):
        _, _, _, y_out = self._forward(X)
        y_pred = np.argmax(y_out, axis=1)

        return y_pred


def plot_data(data, name, style=None, axes=None):
    df = pd.DataFrame({name: data})
    df.plot.line(style=style, ax=axes)


if __name__ == "__main__":

    N = 8

    # lenear regression
    # NOISE = {"loc":0.0, "scale":0.3}
    # plt.figure()
    # lin_ax = plt.subplot(1, 1, 1)
    # noise_data = make_linear_data(N, noise=NOISE)
    # fitted_by_lenear = linear_regression(noise_data)
    # plot_data(fitted_by_lenear, 'fitted by lenear', 'r-', lin_ax)
    # plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', lin_ax)


    # nonlenear regression

    # setup some data
    NOISE = {"loc":0.0, "scale":0.1}
    POLY_PARAM_NUM = 10
    RBF_PARAM_NUM = 4

    plt.figure()
    # 多項式, RBF
    poly_ax = plt.subplot(2, 2, 1)
    poly_reg_ax = plt.subplot(2, 2, 2)
    rbf_ax = plt.subplot(2, 2, 3)
    rbf_reg_ax = plt.subplot(2, 2, 4)
    axes = [poly_ax, poly_reg_ax, rbf_ax, rbf_reg_ax]

    # 多項式のみ
    # poly_ax = plt.subplot(1, 2, 1)
    # poly_reg_ax = plt.subplot(1, 2, 2)
    # axes = [poly_ax, poly_reg_ax]

    # RBFのみ
    rbf_ax = plt.subplot(1, 2, 1)
    rbf_reg_ax = plt.subplot(1, 2, 2)
    axes = [rbf_ax, rbf_reg_ax]

    noise_data = make_sin_data(N, noise=NOISE)
    # regularizer_lambda = np.exp(-18)
    regularizer_lambda = 0.1

    # # polynomial without regularization
    # # fitted_by_poly = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, 0, plot=poly_ax)
    # fitted_by_poly = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, plot=False)
    # plot_data(fitted_by_poly, 'fitted by poly (M={})'.format(POLY_PARAM_NUM), 'r-', poly_ax)
    # # polynomial without regularization
    # # fitted_by_poly_reg = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=poly_reg_ax)
    # fitted_by_poly_reg = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=False)
    # plot_data(fitted_by_poly_reg, 'fitted by poly (M={})'.format(POLY_PARAM_NUM), 'r-', poly_reg_ax)

    # rbf without regularization
    # fitted_by_rbf = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, 0, plot=rbf_ax)
    fitted_by_rbf = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, plot=False)
    plot_data(fitted_by_rbf, 'fitted by rbf (M={})'.format(RBF_PARAM_NUM), 'b-', rbf_ax)
    # rbf without regularization
    # fitted_by_rbf_reg = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=rbf_reg_ax)
    fitted_by_rbf_reg = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=False)
    plot_data(fitted_by_rbf_reg, 'fitted by rbf (M={})'.format(RBF_PARAM_NUM), 'b-', rbf_reg_ax)

    # plotting noisy data
    for ax in axes:
        plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', ax)
    plt.show()
