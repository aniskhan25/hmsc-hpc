import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
tfr, tfm, tfla = tf.random, tf.math, tf.linalg
import tensorflow_probability as tfp
tfd = tfp.distributions


from hmsc.updaters.updateBetaLambda import updateBetaLambda

def _simple_model(has_phylogeny=False, dtype = np.float64):
    # test model - simple integrity

    ny, ns, nc, nt, nr = 701, 7001, 51, 31, 7
    
    mGamma = tfr.normal([nc*nt], dtype=dtype)
    iUGamma = tf.eye(nc*nt, dtype=dtype)
    rhopw = tfr.normal([101,2], dtype=dtype)
    V0 = tf.eye(nc, dtype=dtype)
    f0 = nc + 1

    npVec = np.maximum((ny / (2**np.arange(nr))).astype(int), 2)
    nfVec = 3 + np.arange(nr)
    Pi = np.tile(np.arange(ny)[:,None], [1,nr]) % npVec
    for r in range(nr): np.random.shuffle(Pi[:,r])

    Beta = tfr.normal([nc,ns], dtype=dtype)
    Gamma = tfr.normal([nc,nt], dtype=dtype)
    iV = tfla.inv(tfp.distributions.WishartTriL(tf.cast(f0, dtype), tfla.cholesky(tfla.inv(V0))).sample())
    EtaList = [tfr.normal([npVec[r],nfVec[r]], dtype=dtype) for r in range(nr)]
    LambdaList = [tfr.normal([nfVec[r],ns], dtype=dtype) for r in range(nr)]
    rhoInd = tf.cast(tf.constant([0]), tf.int32)
    sigma = tfr.uniform([ns], dtype=dtype)

    X = np.random.normal(size=[ny,nc])
    Xeff = tf.constant(X, dtype=dtype)
    Y = Z = tf.matmul(X,Beta) + sum([tf.matmul(tf.gather(EtaList[r], Pi[:,r]), LambdaList[r]) for r in range(nr)]) + tfr.normal([ny,ns], 0, sigma, dtype=dtype)
    iD = tf.cast(tfm.logical_not(tfm.is_nan(Y)), dtype) * tf.ones_like(Z) * sigma**-2

    PsiList = [1 + tf.abs(tfr.normal([nfVec[r],ns], dtype=dtype)) for r in range(nr)]
    DeltaList = [1 + tf.abs(tfr.normal([nfVec[r],1], dtype=dtype)) for r in range(nr)]

    T = np.random.normal(size=[ns,nt])

    if has_phylogeny:
        rhoGroup = np.asarray([0] * nc)
        C = np.eye(ns)
        eC, VC = np.linalg.eigh(C)
    else:
        rhoGroup = np.asarray([0] * nc)
        C, eC, VC = None, None, None

    Beta = tf.matmul(Gamma,T,transpose_b=True) + \
        tf.transpose(tfd.MultivariateNormalFullCovariance(covariance_matrix=tfla.inv(iV)).sample(ns))

    params = {}
    modelData = {}
    modelDims = {}
    priorHyperparams = {}
    rLHyperparams = {}

    params["Beta"] = Beta
    params["Gamma"] = Gamma
    params["Lambda"] = LambdaList
    params["Eta"] = EtaList
    params["iV"] = iV
    params["rhoInd"] = rhoInd
    params["Xeff"] = Xeff
    params["Psi"] = PsiList
    params["Delta"] = DeltaList
    params["Z"] = Z
    params["iD"] = iD

    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["ns"] = ns
    modelDims["nr"] = nr
    modelDims["nf"] = nfVec
    modelDims["np"] = npVec

    modelData["Pi"] = Pi
    modelData["T"] = T
    modelData["Yo"] = np.ones((ny, ns), dtype=bool)
    modelData["distr"] = np.ones((ns, 2), dtype=np.int32)
    modelData["rhoGroup"] = rhoGroup
    modelData["C"], modelData["eC"], modelData["VC"] = C, eC, VC

    priorHyperparams["mGamma"] = mGamma
    priorHyperparams["iUGamma"] = iUGamma
    priorHyperparams["rhopw"] = rhopw
    priorHyperparams["V0"] = V0
    priorHyperparams["f0"] = f0

    return params, modelDims, modelData, priorHyperparams, rLHyperparams

#@pytest.mark.parametrize("has_phylogeny", [False, True]) # has_phylogeny=True test is slow; uncomment in final version
@pytest.mark.parametrize("has_phylogeny", [False, False])
def test_updateGammaV(has_phylogeny):

    params, modelDims, modelData, priorHyperparams, _ = _simple_model()

    BetaTrue = params["Beta"]
    LambdaListTrue = params["Lambda"]

    Beta, LambdaList = updateBetaLambda(params, modelData, priorHyperparams)

    #assert_allclose(Beta, BetaTrue, atol=0.1)
    assert_allclose(tf.reduce_mean(Beta), tf.reduce_mean(BetaTrue), atol=.05)

    for r in range(modelDims["nr"]):
        assert_allclose(tf.reduce_mean(LambdaList[r]), tf.reduce_mean(LambdaListTrue[r]), atol=0.005)

def test_updateBetaLambda_shape():

    params, modelDims, modelData, priorHyperparams, _ = _simple_model()
    
    Beta, LambdaList = updateBetaLambda(params, modelData, priorHyperparams)

    assert tf.shape(Beta)[0] == modelDims["nc"]
    assert tf.shape(Beta)[1] == modelDims["ns"]

    for r in range(modelDims["nr"]):
        assert tf.shape(LambdaList[r])[0] == modelDims["nf"][r]
        assert tf.shape(LambdaList[r])[1] == modelDims["ns"]
