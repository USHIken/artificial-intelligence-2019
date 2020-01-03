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

def get_relative_error(actual, measured, percent=True):
    e = abs(actual - measured)/actual
    return e*100 if percent else e

def plot_data(data, name, style=None, axes=None):
    df = pd.DataFrame({name: data})
    df.plot.line(style=style, ax=axes)