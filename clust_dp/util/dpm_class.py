import numpy as np
from clust_dp.util.util_funcs import cholupdowndate, ZZ, plot_cov_ellipse
import matplotlib.pyplot as plt
import copy

COLORS = ['r','b','k','y','c','m']*2


class DPM():
    def __init__(self, KK, alpha, prior, data, z):
        """
        initialize DP mixture model
        :param KK: active mixture components
        :param alpha: concentration parameter
        :param prior: function for Gaussian Wishart prior
        :param data: data as shape [NN,dim]
        :param z: some initial cluster assignments
        """
        self.KK = KK
        self.NN, self.dim = data.shape
        self.alpha = alpha
        self.prior = prior
        self.data = data
        self.z = z
        self.N_k = [0]*KK # Number of points in cluster k, like N_k in Murphy eq25.35 pg 888

        self.qq = []
        # Initialize the priors on the mixture components
        for _ in range(KK):
            self.qq.append(copy.deepcopy(prior))

        # And add items to the mixture components
        for i,x in enumerate(data):
            k = self.z[i]
            self.qq[k].num += 1
            self.qq[k].rr += 1
            self.qq[k].nu += 1
            self.qq[k].sigma_chol = cholupdowndate(self.qq[k].sigma_chol, x, '+')
            self.qq[k].mu_ += x
            self.N_k[k] += 1


    def step(self):
        """
        Make one step in the collapsed Gibbs sampling
        :return:
        """

        # For one Gibbs sample:
        # 1. Remove one x_i from the model (so remove its sufficient statistics from the cluster it is currently assigned to)
        # 2. Make the distro over clusters for this point (eq 25.33 Muprhy pg 888)
        # 3. Sample from this distro and put it in that cluster

        for i,xx in enumerate(self.data):
            ### 1 ###
            k_old = self.z[i]
            self.N_k[k_old] -= 1
            self.qq[k_old].delitem(xx)
            self.remove_cluster_if_empty(k_old)

            ### 2 ###
            pp = self.N_k.copy()
            pp.append(self.alpha)
            pp = np.log(np.array(pp))
            for k in range(self.KK+1):
                pp[k] += self.logpredictive(k,xx)
            pp = np.exp(pp-np.max(pp)) #Subtract max to avoid numerical errors
            pp /= np.sum(pp)

            #Random sample from the conditional probabilities
            # Corresponds to line10 in algorithm25.7 Murphy pg 889
            # k_new = np.random.choice(self.KK+1, p=pp)
            uu = np.random.rand()
            k_new = np.sum(uu>np.cumsum(pp))

            ### 3 ###
            self.add_cluster_maybe(k_new)

            self.z[i] = k_new
            self.N_k[k_new] += 1
            self.qq[k_new].additem(xx)

    def add_cluster_maybe(self, k_new):
        """
        Maybe adds a cluster in case we sample a new k.
        In the Gibbs sample, if you draw z_i=k*, then we add a new cluster.
        This is described by Murphy eq25.38 and the subsequent text
        :param k_new:
        :return:
        """
        if k_new == self.KK:
            self.KK += 1
            self.N_k.append(0)
            self.qq.append(copy.deepcopy(self.prior))


    def logpredictive(self, k, xx):
        """
        Calculates the log predictive distro for x_i given all other x_{-i}
        Corresponds to Murphy eq25.36 pg 888

        Note that if k == KK, then the posterior log-predictive corresponds to the prior log predictive.
        (See eq25.37-38 Murphy pg 888)
        :param k: the cluster under consideration. (Corresponds to 'z_i=k' in eq25.36)
        :param xx: the x_i under consideration
        :return:
        """
        if not k == self.KK:
            q = self.qq[k]
        else:
            q = copy.deepcopy(self.prior)
        return q.logpred(xx)

    def remove_cluster_if_empty(self, k):
        """
        If the cluster k is empty, then remove it.

        For example: Some cluster, k, has one data point, x_i, assigned to it. We make a Gibbs sample and remove
        x_i. Then cluster k is empty and we remove it from our state
        :param k:
        :return:
        """
        if self.N_k[k] == 0:
            self.KK -= 1
            self.qq.pop(k)
            self.N_k.pop(k)
            self.z[np.argwhere(self.z>k)] -= 1

    def plot_data(self, direc = 'im', iter=0):
        f, ax = plt.subplots(1, 1) #PyPlot magic *sarcasm*
        for i in range(self.NN):
            color = COLORS[self.z[i]]
            ax.scatter(self.data[i][0], self.data[i][1], c=color)
        for k in range(self.KK):
            mu, sigma = self.qq[k].get_posterior_NIW(mode='MAP')
            plot_cov_ellipse(sigma, mu, ax=ax, ec=COLORS[k])
        ax.set_title('step' + str(iter).zfill(4))
        plt.savefig(direc+'/step' + str(iter).zfill(4)+'.png')
        plt.close(f)

