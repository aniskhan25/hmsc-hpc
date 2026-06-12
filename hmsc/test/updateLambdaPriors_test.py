import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
tfr, tfm, tfla = tf.random, tf.math, tf.linalg
import tensorflow_probability as tfp
tfd = tfp.distributions


from hmsc.updaters.updateLambdaPriors import updateLambdaPriors

def _simple_model(has_phylogeny=False, dtype = np.float64):

    ny, ns, nc, nt, nr, ncsel = 701, 7001, 51, 31, 7, 50

    mGamma = tfr.normal([nc*nt], dtype=dtype)
    iUGamma = tf.eye(nc*nt, dtype=dtype)
    rhopw = tfr.normal([101,2], dtype=dtype)
    V0 = tf.eye(nc, dtype=dtype)
    f0 = nc + 1

    npVec = np.maximum((ny / (2**np.arange(nr))).astype(int), 2)
    nfVec = 3 + np.arange(nr)
    Pi = np.tile(np.arange(ny)[:,None], [1,nr]) % npVec
    for r in range(nr): np.random.shuffle(Pi[:,r])

    X = np.random.normal(size=[ny,nc])
    T = np.random.normal(size=[ns,nt])

    Gamma = tfr.normal([nc,nt], dtype=dtype)
    iV = tfla.inv(tfp.distributions.WishartTriL(tf.cast(f0, dtype), tfla.cholesky(tfla.inv(V0))).sample())
    Beta = tf.matmul(Gamma,T,transpose_b=True) + \
        tf.transpose(tfd.MultivariateNormalFullCovariance(covariance_matrix=tfla.inv(iV)).sample(ns))
    EtaList = [tfr.normal([npVec[r],nfVec[r]], dtype=dtype) for r in range(nr)]
    LambdaList = [tfr.normal([nfVec[r],ns], dtype=dtype) for r in range(nr)]
    rhoInd = tf.cast(tf.constant([0]), tf.int32)
    sigma = tfr.uniform([ns], dtype=dtype)
    wRRR = tf.zeros([0,0], dtype=dtype)

    XRRR = np.zeros([ny,0], dtype)
    Y = Z = tf.matmul(X,Beta) + sum([tf.matmul(tf.gather(EtaList[r], Pi[:,r]), LambdaList[r]) for r in range(nr)]) + tfr.normal([ny,ns], 0, sigma, dtype=dtype)
    iD = tf.cast(tfm.logical_not(tfm.is_nan(Y)), dtype) * tf.ones_like(Z) * sigma**-2

    XSel = [{} for i in range(ncsel)]
    for i in range(ncsel):
        covGroup = np.array([i+1])
        spGroup = np.arange(ns)
        q = np.random.normal(size=[ns])
        XSel[i]["covGroup"] = covGroup
        XSel[i]["spGroup"] = spGroup
        XSel[i]["q"] = q

    PsiList = [1 + tf.abs(tfr.normal([nfVec[r],ns], dtype=dtype)) for r in range(nr)]
    DeltaList = [1 + tf.abs(tfr.normal([nfVec[r],1], dtype=dtype)) for r in range(nr)]
    
    if has_phylogeny:
        rhoGroup = np.asarray([0] * nc)
        C = np.eye(ns)
        eC, VC = np.linalg.eigh(C)
    else:
        rhoGroup = np.asarray([0] * nc)
        C, eC, VC = None, None, None

    BetaSel = [tf.constant([False] * ns, dtype=tf.bool) for i in range(ncsel)]
    if ncsel > 0:
        bsCovGroupLen = [XSelElem["covGroup"].size for XSelElem in XSel]
        bsInd = tf.concat([XSelElem["covGroup"] for XSelElem in XSel], 0)
        bsActiveList = [tf.gather(BetaSelElem, XSelElem["spGroup"]) for BetaSelElem, XSelElem in zip(BetaSel, XSel)]
        bsActive = tf.cast(tf.repeat(tf.stack(bsActiveList, 0), bsCovGroupLen, 0), dtype)
        bsMask = tf.tensor_scatter_nd_min(tf.ones([nc,ns], dtype), bsInd[:,None], bsActive)
        if X.ndim == 2:
            Xeff = tf.einsum("ik,kj->jik", X, bsMask)
        else:
            Xeff = tf.einsum("jik,kj->jik", X, bsMask)
    else:
        Xeff = X

    params = {}
    modelData = {}
    modelDims = {}
    priorHyperparams = {}

    params["Beta"] = Beta
    params["BetaSel"] = BetaSel
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
    params["wRRR"] = wRRR

    modelDims["ny"] = ny
    modelDims["nc"] = nc
    modelDims["ncsel"] = ncsel
    modelDims["nt"] = nt
    modelDims["ns"] = ns
    modelDims["nr"] = nr
    modelDims["nf"] = nfVec
    modelDims["np"] = npVec

    modelData["X"] = X
    modelData["XRRR"] = XRRR
    modelData["XSel"] = XSel
    modelData["Pi"] = Pi
    modelData["T"] = T
    modelData["rhoGroup"] = rhoGroup
    modelData["C"], modelData["eC"], modelData["VC"] = C, eC, VC

    rLHyperparams = [None] * nr
    for r in range(nr):
        rLPar = {}
        rLPar["sDim"] = 0
        rLPar["xDim"] = 0
        rLPar["nu"] = 3
        rLPar["a1"] = 50
        rLPar["b1"] = 1
        rLPar["a2"] = 50
        rLPar["b2"] = 1
        rLHyperparams[r] = rLPar

    return params, modelDims, modelData, priorHyperparams, rLHyperparams

def test_updateLambdaPriors():
    
    params, modelDims, modelData, _, rLHyperparams = _simple_model()

    PsiListTrue = params["Psi"]
    DeltaListTrue = params["Delta"]

    PsiList, DeltaList = updateLambdaPriors(params, rLHyperparams)

    for r in range(modelDims["nr"]):
        assert_allclose(tf.reduce_mean(PsiList[r]), tf.reduce_mean(PsiListTrue[r]), atol=1.5)
        assert_allclose(tf.reduce_mean(DeltaList[r]), tf.reduce_mean(DeltaListTrue[r]), atol=1.0)

def test_updateLambdaPriors_shape():

    params, modelDims, modelData, _, rLHyperparams = _simple_model()

    PsiList, DeltaList = updateLambdaPriors(params, rLHyperparams)

    for r in range(modelDims["nr"]):
        assert tf.shape(PsiList[r])[0] == rLHyperparams[r]["nu"] + r
        assert tf.shape(PsiList[r])[1] == modelDims["ns"]

        assert tf.shape(DeltaList[r])[0] == rLHyperparams[r]["nu"] + r


def test_updateLambdaPriors_random_slope_shape():
    dtype = np.float64
    params = {
        "Lambda": [tf.ones([2, 3, 2], dtype=dtype)],
        "Psi": [tf.ones([2, 3, 2], dtype=dtype)],
        "Delta": [tf.ones([2, 1], dtype=dtype)],
    }
    rLHyperparams = [
        {
            "xDim": 2,
            "nu": 3,
            "a1": 2,
            "b1": 1,
            "a2": 3,
            "b2": 1,
        }
    ]

    PsiList, DeltaList = updateLambdaPriors(params, rLHyperparams, dtype=dtype)

    assert PsiList[0].shape == (2, 3, 2)
    assert DeltaList[0].shape == (2, 1)
    assert np.isfinite(PsiList[0].numpy()).all()
    assert np.isfinite(DeltaList[0].numpy()).all()
