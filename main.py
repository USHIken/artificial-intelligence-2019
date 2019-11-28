import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

axes = plt.subplot(1, 1, 1)

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


def nonlenear_regression(data, basis_func, param, plot=False):

    if basis_func == polynomial_basis_func:
        param += 1  # j = 1 .. param

    N = data.shape[0]
    x = data.index
    y = data.values

    phi_x = basis_func(x, param)
    phi_x_t = phi_x.T
    W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x)), phi_x_t), y)
    Y = np.zeros(N)
    for j in range(param):
        Y += W[j] * phi_x[:, j]

    # plotting curve
    if plot:
        for j in range(param):
            df = pd.DataFrame(pd.Series(W[j]*phi_x[:, j], index=x))
            df.plot(color='k', legend=None, ax=axes, style="--")

    data = pd.Series(Y, index=data.index)

    return data


if __name__ == "__main__":
    N = 100
    POLY_PARAM_NUM = 3
    RBF_PARAM_NUM = 10
    NOISE = {"loc":0.0, "scale":0.3}
    PLOT = True

    # nonlenear regression
    noise_data = make_sin_data(N, noise=NOISE)
    fitted_by_poly = nonlenear_regression(noise_data, polynomial_basis_func, POLY_PARAM_NUM, PLOT)
    fitted_by_rbf = nonlenear_regression(noise_data, radial_basis_func, RBF_PARAM_NUM, PLOT)

    # plotting fitted and noisy data
    series_dict = {}
    series_dict['noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"])] = noise_data
    series_dict['fitted by poly (M={})'.format(POLY_PARAM_NUM)] = fitted_by_poly
    series_dict['fitted by rbf (M={})'.format(RBF_PARAM_NUM)] = fitted_by_rbf

    df = pd.DataFrame(series_dict)
    df.plot.line(style=['g.', 'r-', 'b-'], ax=axes, title="data")
    plt.show()
