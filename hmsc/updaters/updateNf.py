import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfr = tf.math, tf.random

@tf_named_func("nf")
def updateNf(params, rLHyperparams, it, dtype=np.float64):
#def updateNf_ml(EtaList, LambdaList, PsiList, DeltaList, iter, rLPar, dtype=np.float64):
    """Update latent factors:
    Eta - site loadings,
    Lambda - species loadings,
    Psi - local shrinage species loadings (lambda's prior), and
    Delta - delta global shrinage species loadings (lambda's prior).

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Eta - site loadings
        Beta - species niches
        Psi - local shrinage species loadings (lambda's prior)
        Delta - delta global shrinage species loadings (lambda's prior)
        nu -
        a2 - prior parameters
        b2 -
        nfMin -
        nfMax -
        iter -
    """

    LambdaList = params["Lambda"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]
    EtaList = params["Eta"]
    AlphaIndList = params["AlphaInd"]

    c0 = 1
    c1 = 0.0005
    epsilon = 1e-3  # threshold limit
    prop = 1.00  # proportion of redundant elements within columns
    prob = 1 / tf.exp(c0 + c1 * tf.cast(it, dtype))  # probability of adapting

    nr = len(LambdaList)
    EtaNew, LambdaNew, PsiNew, DeltaNew, AlphaIndNew = [[None] * nr for i in range(5)]
    for r, (Lambda, Psi, Delta, Eta, AlphaInd, rLPar) in enumerate(zip(LambdaList, PsiList, DeltaList, EtaList, AlphaIndList, rLHyperparams)):

        nu = rLPar["nu"]
        a2 = rLPar["a2"]
        b2 = rLPar["b2"]
        nfMin = tf.cast(rLPar["nfMin"], tf.int32)
        nfMax = tf.cast(rLPar["nfMax"], tf.int32)

        if tfr.uniform([], dtype=dtype) < prob:
            nf = tf.shape(Lambda)[0]
            ns = Lambda.shape[1] if len(Lambda.shape.as_list()) == 3 else Lambda.shape[-1]
            np = tf.shape(Eta)[0]
            smallLoadingProp = tf.reduce_mean(tf.cast(tfm.abs(Lambda) < epsilon, dtype=dtype), axis=tf.range(1, tf.rank(Lambda)))
            indRedundant = smallLoadingProp >= prop
            numRedundant = tf.reduce_sum(tf.cast(indRedundant, dtype=tf.int32))

            if nf < nfMin or (nf < nfMax and it > 20 and numRedundant == 0 and tf.reduce_all(smallLoadingProp < 0.995)):
                if len(Lambda.shape.as_list()) == 3:
                    x_dim = Lambda.shape[2]
                    LambdaNew[r] = tf.concat([Lambda, tf.zeros([1, ns, x_dim], dtype=dtype)], 0)
                    PsiNew[r] = tf.concat([Psi, tfr.gamma([1, ns, x_dim], nu/2, nu/2, dtype=dtype)], 0)
                else:
                    LambdaNew[r] = tf.concat([Lambda, tf.zeros([1,ns], dtype=dtype)], 0)
                    PsiNew[r] = tf.concat([Psi, tfr.gamma([1,ns], nu/2, nu/2, dtype=dtype)], 0)
                DeltaNew[r] = tf.concat([Delta, tfr.gamma([1,1], a2, b2, dtype=dtype)], 0)
                EtaNew[r] = tf.concat([Eta, tfr.normal([np,1], dtype=dtype)], 1)
                AlphaIndNew[r] = tf.concat([AlphaInd, tf.zeros([1], tf.int32)], 0)
            elif nf > nfMin and numRedundant > 0:
                indRemain = tf.cast(tf.squeeze(tf.where(tfm.logical_not(indRedundant)), -1), tf.int32)
                # if tf.shape(indRemain)[0] < nfMin:
                #     indRemain = tf.concat([indRemain, nf - 1 - tf.range(nfMin - tf.shape(indRemain)[0])], 0)
                LambdaNew[r] = tf.gather(Lambda, indRemain, axis=0)
                PsiNew[r] = tf.gather(Psi, indRemain, axis=0)
                DeltaNew[r] = tf.gather(Delta, indRemain, axis=0)
                EtaNew[r] = tf.gather(Eta, indRemain, axis=1)
                AlphaIndNew[r] = tf.gather(AlphaInd, indRemain, axis=0)
            else:
                LambdaNew[r], PsiNew[r], DeltaNew[r], EtaNew[r], AlphaIndNew[r] = Lambda, Psi, Delta, Eta, AlphaInd
        else:
            EtaNew[r], LambdaNew[r], PsiNew[r], DeltaNew[r], AlphaIndNew[r] = Eta, Lambda, Psi, Delta, AlphaInd
    return LambdaNew, PsiNew, DeltaNew, EtaNew, AlphaIndNew
