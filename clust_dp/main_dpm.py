import numpy as np
from clust_dp.util.niw_class import NIW
from clust_dp.util.dpm_class import DPM

np.random.seed(564)

if __name__ == "__main__":
    """Make some data"""
    dim = 2 # dimensionality of the data. make 2 to make nice plots
    trueK = 5 # true number of clusters for data generation
    s0 = 3 # Standard deviance of the cluster means in the data generation
    ss = 1 # Standard deviance of the data per cluster in the data generation
    # s0 and ss together make the relative variance of the means compared to the data (cluster separability)
    NN = 100 # number of data points

    #Construct a dataset
    true_z = [z for z in range(trueK) for _ in range(int(NN/trueK))]
    mu = np.random.randn(dim,trueK)*s0
    data = (mu[:,true_z] + np.random.randn(dim,NN)*ss).T

    """Set up the priors on the data"""
    q0 = NIW(dim=dim, s0=3, ss=1, nu=3, mu_prior = np.zeros((dim,)))

    """Set up a Mixture model for the Dirichlet process"""
    alpha = 1 # concentration parameter for the DP process. See Murphy eq25.17 pg 884
    KK = 1 # initial guesses to make for number of clusters
    #random initial assignments
    z = np.random.randint(0,KK,(NN,))
    dpm = DPM(KK, alpha, q0, data, z)

    """Start the trial"""
    numiter = 300
    record = []
    for iter in range(numiter):
        dpm.step()
        print('At step %4i/%4i KK is %i'%(iter, numiter, len(dpm.N_k)))
        record.append(len(dpm.N_k)) #Keep track of the number of clusters
        if iter%10 == 0:
            dpm.plot_data(iter=iter)

    print(record)

# convert -delay 20 -loop 0 *.png dpm.gif





