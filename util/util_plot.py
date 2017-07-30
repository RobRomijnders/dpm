import numpy as np
import matplotlib.pyplot as plt

def plot_dp(X,clust,thetas):
    N,D = X.shape
    assert len(clust) == N

    colors = ['r','b']

    x1min, x2min = np.min(X,0)
    x1max, x2max = np.max(X,0)

    plt.figure()
    # plt.axes([x1min, x1max, x2min, x2max])

    for n in range(N):
        plt.scatter(X[n,0],X[n,1],c=colors[clust[n]])

    plt.axes([x1min, x1max, x2min, x2max])
    plt.show()

