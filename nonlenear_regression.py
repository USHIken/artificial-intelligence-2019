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


def nonlenear_regression(train_data, test_data, basis_func, param, regularizer_lambda=0, plot=None):
    """
    非線形回帰を行った結果を返す
    """


    # training
    if basis_func == polynomial_basis_func:
        param += 1  # j = 1 .. param

    x = train_data.index
    y = train_data.values

    phi_x = basis_func(x, param)
    phi_x_t = phi_x.T
    regularizer = regularizer_lambda*np.eye(phi_x.shape[1]) # calc λI for regularization
    # W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x) + regularizer), phi_x_t), y)
    W = np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x) + regularizer), np.dot(phi_x_t, y))

    # plotting curve
    if plot:
        for j in range(param):
            df = pd.DataFrame(pd.Series(W[j]*phi_x[:, j], index=x))
            df.plot(color='k', legend=None, ax=plot, style="--")

    # testing
    test_N = test_data.shape[0]
    test_x = test_data.index
    phi_x = basis_func(test_x, param)
    Y = np.zeros(test_N)
    for j in range(param):
        Y += W[j] * phi_x[:, j]

    data = pd.Series(Y, index=test_x)

    return data


def get_axes(spec):
    poly_ax = None
    poly_reg_ax = None
    rbf_ax = None
    rbf_reg_ax = None

    if spec == "both":      # 多項式, RBF        
        poly_ax = plt.subplot(2, 2, 1)
        poly_reg_ax = plt.subplot(2, 2, 2)
        rbf_ax = plt.subplot(2, 2, 3)
        rbf_reg_ax = plt.subplot(2, 2, 4)

    elif spec == "poly":    # 多項式のみ
        poly_ax = plt.subplot(1, 2, 1)
        poly_reg_ax = plt.subplot(1, 2, 2)

    elif spec == "rbf":     # RBFのみ        
        rbf_ax = plt.subplot(1, 2, 1)
        rbf_reg_ax = plt.subplot(1, 2, 2)
    
    return poly_ax, poly_reg_ax, rbf_ax, rbf_reg_ax

def regularizing_exp():

    plt.figure()
    ax = plt.subplot(1,1,1)

    NOISE = {"loc":0.0, "scale":0.3}
    train_N = 100
    test_N = 1000
    noise_data = make_sin_data(train_N, noise=NOISE)
    test_data = make_sin_data(test_N, noise=False, width=1)
    regularizer_lambda = 1
    RBF_PARAM_NUMs = [3, 5, 10, 20, 30, 50]
    axes = [plt.subplot(2,3,i+1) for i in range(len(RBF_PARAM_NUMs))]
    alphs = list("abcdef")

    for RBF_PARAM_NUM, ax, alph in zip(RBF_PARAM_NUMs, axes, alphs):
        plt.axes(ax)
        # rbf without regularization
        fitted_by_rbf = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, plot=False)
        plot_data(fitted_by_rbf, 'without regularizer', 'r:', ax)
        # rbf with regularization
        fitted_by_rbf_reg = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=False)
        plot_data(fitted_by_rbf_reg, 'with regularizer', 'r-', ax)

        plot_data(test_data, 'test data', 'b--', ax)
        plot_data(noise_data, 'noise data', 'g.', ax)
        plt.title("({}) n={}".format(alph, RBF_PARAM_NUM))
    
    plt.show()


def plotting_exp(spec):

    plt.figure()

    poly, rbf = False, False
    if spec == "both":
        poly, rbf = True, True
    elif spec == "poly":
        poly = True
    elif spec == "rbf":
        rbf = True
    axes = get_axes(spec)
    poly_ax, poly_reg_ax, rbf_ax, rbf_reg_ax = axes

    train_N = 100
    test_N = 10000

    # setup some data
    NOISE = {"loc":0.0, "scale":0.1}
    POLY_PARAM_NUM = 10
    RBF_PARAM_NUM = 30

    noise_data = make_sin_data(train_N, noise=NOISE)
    test_data = make_sin_data(test_N, noise=False, width=1)
    # regularizer_lambda = np.exp(-18)
    regularizer_lambda = 1

    if poly:
        # polynomial without regularization
        fitted_by_poly = nonlenear_regression(noise_data, test_data, polynomial_basis_func, POLY_PARAM_NUM, 0, plot=poly_ax)
        # fitted_by_poly = nonlenear_regression(noise_data, test_data, polynomial_basis_func, POLY_PARAM_NUM, plot=False)
        plot_data(fitted_by_poly, 'poly (M={})'.format(POLY_PARAM_NUM), 'r-', poly_ax)
        # polynomial with regularization
        fitted_by_poly_reg = nonlenear_regression(noise_data, test_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=poly_reg_ax)
        # fitted_by_poly_reg = nonlenear_regression(noise_data, test_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=False)
        plot_data(fitted_by_poly_reg, 'poly with regularizer(M={})'.format(POLY_PARAM_NUM), 'r-', poly_reg_ax)

    if rbf:
        # rbf without regularization
        fitted_by_rbf = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, plot=rbf_ax)
        # fitted_by_rbf = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, plot=False)
        plot_data(fitted_by_rbf, 'rbf (M={})'.format(RBF_PARAM_NUM), 'b-', rbf_ax)
        # rbf with regularization
        fitted_by_rbf_reg = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=rbf_reg_ax)
        # fitted_by_rbf_reg = nonlenear_regression(noise_data, test_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=False)
        plot_data(fitted_by_rbf_reg, 'rbf with regularizer(M={})'.format(RBF_PARAM_NUM), 'b-', rbf_reg_ax)

    # plotting noisy data
    for ax in axes:
        if ax:
            plt.axes(ax)
            plt.ylim(-2, 2)
            plot_data(test_data, 'test data'.format(NOISE["loc"], NOISE["scale"]), 'b--', ax)
            plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', ax)
    plt.show()

def plotting_sin(width):
    x = np.linspace(-width*np.pi, width*np.pi, 201)
    plt.plot(x, np.sin(x))
    plt.xlabel('Angle [rad]')
    plt.ylabel('sin(x)')
    plt.axis('tight')
    plt.show()

if __name__ == "__main__":
    # plotting_exp("rbf")
    regularizing_exp()
    # plotting_sin(2)
