import numpy as np
from scipy.special import gammaln
from scipy.stats import invwishart,multivariate_normal
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#Note: theres two ways to do the Rank1 update to the Cholesky decomposition
# TODO: check which is faster
def cholupdowndate_1(R,x,sign = '+'):
    p = np.size(x)
    x = x.T.copy()
    R=R.copy()
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r

        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return np.triu(R)

def cholupdowndate(R,x,sign = '+'):
    V = R.T.dot(R)
    outer = np.outer(x,x)
    if sign == '+':
        W = V + outer
    elif sign == '-':
        W = V - outer
    return np.linalg.cholesky(W).T


def ZZ(dim,num,rr,nu,sigma_chol,mu_):
    # Some normalization for the Wishart prior in log domain
    # See Murphy eq4.160 pg 128\
    zz = -num * dim/2. * np.log(np.pi) - dim/2. * np.log(rr)- nu * np.sum(np.log(np.diagonal(cholupdowndate(sigma_chol, mu_/np.sqrt(rr), '-')))) + np.sum(gammaln((nu-np.arange(0,dim))/2.))
    return zz


def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)