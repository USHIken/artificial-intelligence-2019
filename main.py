import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_linear_data(n, a=1, b=0, noise={"loc":0.0, "scale":0.1}):
    func = lambda x: a*x + b
    data = make_data(n, func, -10, 10, noise)
    return data


def make_sin_data(n, noise={"loc":0.0, "scale":0.1}):
    func = np.sin
    data = make_data(n, func, -1*np.pi, 1*np.pi, noise)
    return data


def make_data(n, func, start, end, noise):
    x = np.linspace(start, end, n)
    data = pd.Series(func(x), index=x)
    if noise:
        noise_func = lambda data, n: data + np.random.normal(noise["loc"], noise["scale"], n)
        data = noise_func(data, n)
    return data


def linear_regression(data):

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

    N = x.shape[0]

    phi_x = np.zeros((N, param))
    for j in range(param):
        phi_x[:,j] = pow(x, j)
    
    return phi_x
    

def radial_basis_func(x, param, nu=10):

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


def nonlenear_regression(data, basis_func, param, regularizer=0, plot=None):

    if basis_func == polynomial_basis_func:
        param += 1  # j = 1 .. param

    N = data.shape[0]
    x = data.index
    y = data.values

    phi_x = basis_func(x, param)
    phi_x_t = phi_x.T
    regularizer = regularizer_lambda*np.eye(phi_x.shape[1]) # calc λI for regularization
    W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x) + regularizer), phi_x_t), y)
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


def plot_data(data, name, style, axes):
    df = pd.DataFrame({name: data})
    df.plot.line(style=style, ax=axes)


if __name__ == "__main__":

    N = 100

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
    POLY_PARAM_NUM = 3
    RBF_PARAM_NUM = 10

    plt.figure()
    poly_ax = plt.subplot(2, 2, 1)
    poly_reg_ax = plt.subplot(2, 2, 2)
    rbf_ax = plt.subplot(2, 2, 3)
    rbf_reg_ax = plt.subplot(2, 2, 4)
    axes = [poly_ax, poly_reg_ax, rbf_ax, rbf_reg_ax]

    noise_data = make_sin_data(N, noise=NOISE)
    # regularizer_lambda = np.exp(-18)
    regularizer_lambda = 0.001

    # polynomial without regularization
    fitted_by_poly = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, 0, plot=poly_ax)
    # fitted_by_poly = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, plot=False)
    plot_data(fitted_by_poly, 'fitted by poly (M={})'.format(POLY_PARAM_NUM), 'r-', poly_ax)
    # polynomial without regularization
    fitted_by_poly_reg = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=poly_reg_ax)
    # fitted_by_poly_reg = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, regularizer_lambda, plot=False)
    plot_data(fitted_by_poly_reg, 'fitted by poly (M={})'.format(POLY_PARAM_NUM), 'r-', poly_reg_ax)

    # rbf without regularization
    fitted_by_rbf = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, 0, plot=rbf_ax)
    # fitted_by_rbf = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, plot=False)
    plot_data(fitted_by_rbf, 'fitted by rbf (M={})'.format(RBF_PARAM_NUM), 'b-', rbf_ax)
    # rbf without regularization
    fitted_by_rbf_reg = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=rbf_reg_ax)
    # fitted_by_rbf_reg = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, regularizer_lambda, plot=False)
    plot_data(fitted_by_rbf_reg, 'fitted by rbf (M={})'.format(RBF_PARAM_NUM), 'b-', rbf_reg_ax)

    # plotting noisy data
    for ax in axes:
        plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', ax)
    plt.show()
