from utils import *
import pandas as pd
import matplotlib.pyplot as plt


def util_mlp():
    N = 10
    train = make_sin_data(N)
    test = make_sin_data(N)
    mlp = MultilayerPerceptron(epochs=1)
    mlp.fit(train.index, train.values)
    valid_y = mlp.predict(np.array([test.index]).T)
    valid = pd.Series(valid_y, index=test.index)

    train_ax = plt.subplot(1,2,1)
    valid_ax = plt.subplot(1,2,2)
    # axes = [train_ax, valid_ax]
    plot_data(train, "train", axes=train_ax)
    plot_data(valid, "valid", axes=valid_ax)
    plt.show()

if __name__ == "__main__":
    # TODO: adaptive learning late
    util_mlp()
    pass
