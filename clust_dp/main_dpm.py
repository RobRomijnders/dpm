import numpy as np
from clust_dp.util.niw_class import NIW
from clust_dp.util.dpm_class import DPM

np.random.seed(123)

def generate_dataset(N, trueK, mu, ss, dim=2):
    """
    Generates N datapoints with trueK clusters
    :param N: Number of data points
    :param trueK: Number of clusters in the generated data
    :param mu: True means per cluster
    :param ss: Standard deviation of the data per each cluster
    :param dim: dimensionality of the data space
    :return: dataset in [N,dim
    """
    #Construct a dataset
    true_z = [i%trueK for i in range(N)]
    data = (mu[:,true_z] + np.random.randn(dim,N)*ss).T
    return data

def add_points_to_dpm(dpm, data, K):
    K_guess = min(K,len(dpm.qq))
    initial_assignments = np.random.randint(0,K_guess,(data.shape[0]))
    dpm.include_points(data,initial_assignments)
    dpm.z = np.concatenate((dpm.z,initial_assignments),0 )
    dpm.data = np.concatenate((dpm.data,data),0 )


if __name__ == "__main__":
    """Make some data"""
    dim = 2 # dimensionality of the data. make 2 to make nice plots
    trueK = 7 # true number of clusters for data generation
    s0 = 4 # Standard deviance of the cluster means in the data generation
    ss = 1.0 # Standard deviance of the data per cluster in the data generation
    # s0 and ss together make the relative variance of the means compared to the data (cluster separability)
    N = 100 # number of data points


    mu = np.random.randn(dim,trueK)*s0
    data = generate_dataset(N,trueK,mu,ss,dim)

    """Set up the priors on the data"""
    q0 = NIW(dim=dim, s0=3, ss=1, nu=3, mu_prior = np.zeros((dim,)))

    """Set up a Mixture model for the Dirichlet process"""
    alpha = 1 # concentration parameter for the DP process. See Murphy eq25.17 pg 884
    KK = 1 # initial guesses to make for number of clusters
    #random initial assignments
    z = np.random.randint(0,KK,(N,))
    dpm = DPM(KK, alpha, q0, data, z)

    """Start the trial"""
    numiter = 600
    record = []
    for iter in range(1,numiter):
        dpm.step()

        print('At step %4i/%4i KK is %i'%(iter, numiter, len(dpm.N_k)))
        record.append(len(dpm.N_k)) #Keep track of the number of clusters
        if iter%10 == 0:
            dpm.plot_data(iter=iter)
        if iter%50 == 0:
            new_data = generate_dataset(30, trueK, mu, ss, dim)
            add_points_to_dpm(dpm, new_data,KK)

    print(record)

# convert -delay 50 -loop 0 *.png dpm.gif





