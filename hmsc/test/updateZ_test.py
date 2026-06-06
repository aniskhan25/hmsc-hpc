import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
tfr, tfm = tf.random, tf.math

from hmsc.updaters.updateZ import updateZ

def _simple_model(spatial_method="None", dtype = np.float64):

    ny, ns, nc, nf, nr = 701, 1001, 31, 17, 7
    
    npVec = np.maximum((ny / (2**np.arange(nr))).astype(int), 2)
    nfVec = 3 + np.arange(nr)
    Pi = np.tile(np.arange(ny)[:,None], [1,nr]) % npVec
    for r in range(nr): np.random.shuffle(Pi[:,r])

    Beta = tfr.normal([nc,ns], dtype=dtype)
    EtaList = [tfr.normal([npVec[r],nfVec[r]], dtype=dtype) for r in range(nr)]
    LambdaList = [tfr.normal([nfVec[r],ns], dtype=dtype) for r in range(nr)]
    AlphaIndList = [tf.zeros([nfVec[r],1], dtype=tf.int64) for r in range(nr)]
    sigma = tfr.uniform([ns], dtype=dtype)

    X = np.random.normal(size=[ny,nc])
    Xeff = tf.constant(X, dtype=dtype)
    Y = Z = tf.matmul(X,Beta) + sum([tf.matmul(tf.gather(EtaList[r], Pi[:,r]), LambdaList[r]) for r in range(nr)]) + tfr.normal([ny,ns], 0, sigma, dtype=dtype)
    iD = tf.cast(tfm.logical_not(tfm.is_nan(Y)), dtype) * tf.ones_like(Z) * sigma**-2
    distr = np.ones([ns,2])
     
    params = {}
    modelData = {}
    modelDims = {}
    priorHyperparams = {}
    rLHyperparams = {}

    params["Z"] = Z
    params["Beta"] = Beta
    params["Eta"] = EtaList
    params["Lambda"] = LambdaList
    params["AlphaInd"] = AlphaIndList
    params["sigma"] = sigma
    params["iD"] = iD
    params["Xeff"] = Xeff

    modelDims["ny"] = ny
    modelDims["ns"] = ns
    modelDims["nr"] = nr
    modelDims["np"] = npVec
    modelDims["nf"] = nfVec  

    modelData["Pi"] = Pi
    modelData["Y"] = Y
    modelData["Yo"] = tfm.logical_not(tfm.is_nan(Y))
    modelData["distr"] = distr

    rLHyperparams = [None] * nr
    for r in range(nr):
        rLPar = {}
        rLPar["sDim"] = 0
        rLPar["xDim"] = 0
        rLHyperparams[r] = rLPar

    return params, modelDims, modelData, priorHyperparams, rLHyperparams

def test_updateZ():

    params, modelDims, modelData, _, rLHyperparams = _simple_model()

    ZTrue = params["Z"]

    Z, _, _ = updateZ(params, modelData, rLHyperparams, poisson_preupdate_z=False)

    assert_allclose(Z, ZTrue, atol=0.1)

    assert_allclose(tf.reduce_mean(Z), tf.reduce_mean(ZTrue), atol=0.001)

def test_updateZ_shape():

    params, modelDims, modelData, _, rLHyperparams = _simple_model()
    
    Z, _, _ = updateZ(params, modelData, rLHyperparams, poisson_preupdate_z=False)

    assert tf.shape(Z)[0] == modelDims["ny"]
    assert tf.shape(Z)[1] == modelDims["ns"]
