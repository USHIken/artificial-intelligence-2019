import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_linear_data(n, a=1, b=0, noise=True):
    func = lambda x: a*x + b
    data = make_data(n, func, -10, 10, noise)
    return data


def make_sin_data(n, noise=True):
    func = np.sin
    data = make_data(n, func, -1*np.pi, 1*np.pi, noise)
    return data


def make_data(n, func, start, end, noise):
    x = np.linspace(start, end, n)
    data = pd.Series(func(x), index=x)
    if noise:
        noise_func = lambda data, n: data + np.random.normal(0.0, 0.3, n)
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


def nonlenear_regression(data, param):

    N = data.shape[0]
    x = data.index
    y = data.values
    phi_x = np.zeros((N, param+1))
    Y = np.zeros(N)

    for j in range(param+1):
        phi_x[:,j] = pow(x, j)
    phi_x_t = phi_x.T
    W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x)), phi_x_t), y)
    for j in range(param+1):
        Y += W[j] * pow(x, j)

    data = pd.Series(Y, index=x)
    return data


if __name__ == "__main__":
    N = 100

    # noise_data = make_linear_data(N, b=3)
    # data = linear_regression(noise_data)

    noise_data = make_sin_data(N)
    data = nonlenear_regression(noise_data, 3)

    df = pd.DataFrame({'fitted': data, 'noise': noise_data})
    df.plot(color=('r', 'b'))
    plt.show()
