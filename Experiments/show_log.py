import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_path = './NO_STREET_SIE'
    log = np.load(os.path.join(data_path, 'log.npy'))

    plt.plot(log[:, 0], c='r')  # train loss
    plt.plot(log[:, 1], c='g')  # val loss
    plt.plot(log[:, 2], c='b')  # val rmse
    plt.show()


if __name__ == "__main__":
    main()
