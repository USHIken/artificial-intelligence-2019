from utils import *
import pandas as pd
import matplotlib.pyplot as plt


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
    return (data, a, b)

def plotting_exp():

    a = 3
    b = 5
    NOISE = {"loc":0.0, "scale":4}

    plt.figure()

    lin_axes = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)]
    Ns = [10, 50, 100, 500]
    alphs = ["a", "b", "c", "d"]
    
    for N, lin_ax, alph in zip(Ns, lin_axes, alphs):
        h, w, n = lin_ax
        lin_ax = plt.subplot(h, w, n)
        noise_data = make_linear_data(N, a, b, noise=NOISE)
        clean_data = make_linear_data(N, a, b, noise=False)
        fitted_by_lenear, fitted_a, fitted_b = linear_regression(noise_data)
        plot_data(clean_data, 'clean data (a={:.2f}, b={:.2f})'.format(a, b), 'b--', lin_ax)
        plot_data(fitted_by_lenear, 'fitted data (a={:.2f}, b={:.2f})'.format(fitted_a, fitted_b), 'r-', lin_ax)
        plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', lin_ax)
        error_a = get_relative_error(a, fitted_a)
        error_b = get_relative_error(b, fitted_b)
        plt.title("({}) N={}, error: a->{:.2f}%, b->{:.2f}%".format(alph, N, error_a, error_b))
    
    plt.show()

def measuring_error_exp():

    a = 3
    b = 5
    NOISE = {"loc":0.0, "scale":4}
    N_max = 500

    iter_num = 1000
    error_a_aves = np.array([])
    error_b_aves = np.array([])
    error_a_vars = np.array([])
    error_b_vars = np.array([])
    n_idx = np.arange(N_max)[2:]

    plt.figure()

    for N in n_idx:
        noise_data = make_linear_data(N, a, b, noise=NOISE)
        error_a = np.array([])
        error_b = np.array([])

        for _ in range(iter_num):
            _, fitted_a, fitted_b = linear_regression(noise_data)
            error_a = np.append(error_a, get_relative_error(a, fitted_a))
            error_b = np.append(error_b, get_relative_error(b, fitted_b))

        error_a_aves = np.append(error_a_aves, np.mean(error_a))
        error_b_aves = np.append(error_b_aves, np.mean(error_b))
        error_a_vars = np.append(error_a_vars, np.var(error_a))
        error_b_vars = np.append(error_b_vars, np.var(error_b))
        # print("N={}, error_a={:.2f}, error_b={:.2f}".format(N, error_a/iter_num, error_b/iter_num))
        print("{}/{}".format(N, N_max))
    error_a_aves = pd.Series(error_a_aves, n_idx)
    error_b_aves = pd.Series(error_b_aves, n_idx)
    error_a_vars = pd.Series(error_a_vars, n_idx)
    error_b_vars = pd.Series(error_b_vars, n_idx)
    print(error_a)
    ax_ave = plt.subplot(1, 2, 1)
    plt.xlabel("N[-]")
    plt.ylabel("error[%]")
    plot_data(error_a_aves, 'error a', 'r-', ax_ave)
    ax_var = plt.subplot(1, 2, 2)
    plt.xlabel("N [-]")
    plt.ylabel("error [%]")
    plot_data(error_b_aves, 'error b', 'b-', ax_var)
    # plot_data(error_a_vars, 'error_a', 'r-', ax_var)
    # plot_data(error_b_vars, 'error_b', 'b--', ax_var)

    plt.show()


if __name__ == "__main__":
    # plotting_exp()
    measuring_error_exp()
    pass