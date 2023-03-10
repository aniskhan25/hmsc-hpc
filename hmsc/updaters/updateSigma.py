import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tf.math


def updateSigma(params, data, priorHyperparameters, dtype=np.float64):
    """Update prior(s) for whole model:
    sigma - residual variance.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Z - latent variables
        Beta - species niches
        Eta - site loadings
        Lambda - species loadings
        sigma - residual variance
        Y - community data
        X - environmental data
        Pi - study design
        distr -
        aSigma -
        bSigma -
    """

    Z = params["Z"]
    Beta = params["Beta"]
    EtaList = params["Eta"]
    LambdaList = params["Lambda"]
    sigma = params["sigma"]

    Y = data["Y"]
    X = data["X"]
    Pi = data["Pi"]
    distr = data["distr"]
    aSigma = priorHyperparameters["aSigma"]
    bSigma = priorHyperparameters["bSigma"]

    nr = len(EtaList)
    indVarSigma = tf.cast(tf.equal(distr[:,1], 1), dtype)
    LFix = tf.matmul(X, Beta)
    LRanLevelList = [None] * nr
    for r, (Eta, Lambda) in enumerate(zip(EtaList, LambdaList)):
        LRanLevelList[r] = tf.matmul(tf.gather(Eta, Pi[:, r]), Lambda)

    L = LFix + sum(LRanLevelList)
    Eps = Z - L

    alpha = aSigma + Y.shape[0] / 2.0
    beta = bSigma + tf.reduce_sum(Eps**2, axis=0) / 2.0
    isigma2 = tfd.Gamma(concentration=alpha, rate=beta).sample()
    sigmaNew = indVarSigma * tfm.rsqrt(isigma2) + (1 - indVarSigma) * sigma

    return sigmaNew
