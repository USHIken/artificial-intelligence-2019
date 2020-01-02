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
    return data


if __name__ == "__main__":

    N = 8

    # lenear regression
    NOISE = {"loc":0.0, "scale":0.3}
    plt.figure()
    lin_ax = plt.subplot(1, 1, 1)
    noise_data = make_linear_data(N, noise=NOISE)
    fitted_by_lenear = linear_regression(noise_data)
    plot_data(fitted_by_lenear, 'fitted by lenear', 'r-', lin_ax)
    plot_data(noise_data, 'noise (loc={}, scale={})'.format(NOISE["loc"], NOISE["scale"]), 'g.', lin_ax)
    plt.show()