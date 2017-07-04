import numpy as np
from clust_dp.util.util_funcs import cholupdowndate, ZZ, plot_cov_ellipse
from scipy.stats import invwishart,multivariate_normal

class NIW():
    def __init__(self, dim, s0, ss, nu, mu_prior):
        """
        A Normal Inverse Wishart prior see Murphy eq 4.199 pg 135
        :param dim: dimensionality of the data. make 2 to make nice plots
        :param s0: Standard deviance of the cluster means in the data generation
        :param ss: Standard deviance of the data per cluster in the data generation.
        s0 and ss together make the relative variance of the means compared to the data (cluster separability)
        :param nu: degrees of freedom of inverse Wishart covariance prior. Corresponds to nu notation in Murphy chap 4.5 starting pg 128
        :param mu_prior: prior mean vector
        """
        self.dim = dim
        self.num = 0 # number of points currently assigned to the cluster
        self.hhss = s0**2/ss**2
        self.rr = rr = 1./self.hhss # relative precision of self.mu_
        self.nu = nu
        SS = 2 * ss**2 * np.eye(dim) * nu
        self.sigma_chol = np.linalg.cholesky(SS + rr * np.outer(mu_prior,mu_prior)).T #Do .T to make upper triangular
        self.mu_ = rr*mu_prior  #Note that this one is unnormalized
        self.Z0 = ZZ(dim,self.num,rr,nu,self.sigma_chol,self.mu_)

    def logpred(self, xx):
        numerator = ZZ(self.dim, self.num+1, self.rr+1, self.nu+1, cholupdowndate(self.sigma_chol, xx), self.mu_+xx)
        denominator = ZZ(self.dim, self.num, self.rr, self.nu, self.sigma_chol, self.mu_)
        result = numerator - denominator # In log domain subtract them
        return result
    def delitem(self, xx):
        """
        Deletes the sufficient statistics of xx from the Gaussian Wishart
        :return:
        """
        self.num -= 1
        self.rr -= 1
        self.nu -= 1
        self.sigma_chol = cholupdowndate(self.sigma_chol, xx, '-')
        self.mu_ -= xx
    def additem(self, xx):
        """
        Adds the sufficient statistics of xx from the Gaussian Wishart
        :return:
        """
        self.num += 1
        self.rr += 1
        self.nu += 1
        self.sigma_chol = cholupdowndate(self.sigma_chol, xx, '+')
        self.mu_ += xx
    def get_posterior_NIW(self, mode='sample'):
        """
        Gets a sample from the posterior NIW distro
        Normal-Inverse-Wishart distro is described in Murphy eq4.200 pg 135
        :param qq:
        :param mode: either 'sample' from the posterior or get the 'MAP'
        :return:
        """
        C_upper = cholupdowndate(self.sigma_chol, self.mu_/np.sqrt(self.rr), '-')
        C = C_upper.T.dot(C_upper)

        mean = self.mu_/self.rr
        if mode == 'sample':
            sigma = invwishart.rvs(scale=C, df=self.nu)
            mu = multivariate_normal.rvs(mean, sigma/self.rr)
        elif mode == 'MAP':
            sigma = C/(self.nu - self.dim -1.)
            mu = mean
        return mu, sigma

    def logmarginal(self):
        ll = ZZ(self.dim, self.num, self.rr, self.nu, self.sigma_chol, self.mu_) - self.Z0
        return ll

