import numpy as np
from scipy.stats import wishart, uniform,multivariate_normal
from clust_dp.util.util_plot import plot_dp

N = 1000
alpha = 4

class H_measure():
    def __init__(self):
        # Wishart distributes over covariance matrices (PSD matrices in general)
        self.wishart = wishart(df = 2, scale=[[1.,.5],[.5,1.]])
        #Have our means in the rectangle (x=0,y=1)+(dx=5,dy=4)
        self.unif1 = uniform(0,5)
        self.unif2 = uniform(1,4)

    def sample(self):
        mu = np.array([self.unif1.rvs(), self.unif2.rvs()])
        sigma = self.wishart.rvs()
        return mu, sigma

class DP():
    def __init__(self, alpha, H):
        self.alpha = alpha
        self.H = H

        self.K = 0  #The current number of clusters in our process

        #initialized on the first sample:
        self.Nk = [0] #List of current numbers per cluster
        self.theta = [] #List of parameters per cluster

    def sample(self):
        """
        Samples from the Dirichlet process via a Chinese Restaurant process
        - See Murphy2012 pg. 886, eq 25.28
        - See http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/

        :return: theta  : (mean, sigma)
                  k     : the index of cluster. Might be used for plotting.
        """

        if self.K == 0:
            return self.first_sample()
        N = sum(self.Nk)
        pred_distr = []
        for nk in self.Nk:
            pred_distr.append(nk/(self.alpha + N)) #Could do without for loop, but this is more explicit
        pred_distr.append(self.alpha/(self.alpha+N))
        assert np.abs(sum(pred_distr)-1.) < 1E-5
        k = np.random.choice(self.K+1,p=pred_distr) #Samples from 0:K with bucket probabilities pred_distr
        if k < self.K:
            return self.theta[k],k
        else:
            self.theta.append(self.H.sample())
            return self.theta[-1],k

    def first_sample(self):
        self.Nk[0] += 1
        self.theta.append(self.H.sample())
        self.K = 1
        return self.theta[0],0


def non_parametric_mixture(dp, N = 100):
    # Draws N samples from the mixture model defined by the Dirichlet Process
    X = []
    clusters = []
    for n in range(N):
        (mu, sigma), k = dp.sample()
        x = multivariate_normal.rvs(mean=mu, cov=sigma)
        X.append(x)
        clusters.append(k)
    return X, clusters, dp.theta



def main():
    H = H_measure()
    dp = DP(alpha=1, H=H)
    X, clust, thetas = non_parametric_mixture(dp,N=300)
    plot_dp(np.stack(X),np.array(clust),thetas)
    pass



if __name__ == "__main__":
    main()