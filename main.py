import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

axes = plt.subplot(1, 1, 1)

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
        noise_func = lambda data, n: data + np.random.normal(0.0, 0.1, n)
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


def polynomial_basis_func(data, param):

    param += 1
    N = data.shape[0]
    x = data.index
    y = data.values

    phi_x = np.zeros((N, param))
    Y = np.zeros(N)

    for j in range(param):
        phi_x[:,j] = pow(x, j)
    phi_x_t = phi_x.T
    W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x)), phi_x_t), y)
    for j in range(param):
        Y += W[j] * phi_x[:, j]
    
    return Y
    

def radial_basis_func(data, param, nu=2):

    N = data.shape[0]
    x = data.index
    y = data.values

    A = np.array_split(x, param)

    c = [0]*param
    for j in range(param):
        c[j] = sum(A[j]) / len(A[j])

    s2 = [0]*param
    for j in range(param):
        s2[j] = np.linalg.norm(A[j] - c[j], ord=2)**2 / len(A[j])

    phi_x = np.ones((N, param+1))
    Y = np.zeros(N)
    for j in range(param):
        phi_x[:,j+1] = np.exp(-((x-c[j])**2) / (nu * s2[j]))
    phi_x_t = phi_x.T
    W = np.dot(np.dot(np.linalg.inv(np.dot(phi_x_t, phi_x)), phi_x_t), y)
    for j in range(param):
        Y += W[j] * phi_x[:, j]

    # plotting RBF
    for j in range(1,param):
        df = pd.DataFrame(pd.Series(W[j]*phi_x[:, j], index=x))
        df.plot(color='k', legend=None, ax=axes)
    
    return Y


def nonlenear_regression(data, param):
    #Y = polynomial_basis_func(data, param)
    Y = radial_basis_func(data, param, nu=10)
    data = pd.Series(Y, index=data.index)
    return data


if __name__ == "__main__":
    N = 100

    # lenear regression
    # noise_data = make_linear_data(N, b=3)
    # data = linear_regression(noise_data)

    # nonlenear regression
    noise_data = make_sin_data(N)
    data = nonlenear_regression(noise_data, 10)

    # plotting fitted and noisy data
    df = pd.DataFrame({'fitted data': data, 'noisy data': noise_data})
    df.plot.line(style=['r-', 'b.'], ax=axes)
    plt.show()
