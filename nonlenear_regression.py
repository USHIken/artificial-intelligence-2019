from utils import *
import pandas as pd
import matplotlib.pyplot as plt


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


if __name__ == "__main__":

    N = 100

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
