import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import tensorflow as tf
tfr, tfm = tf.random, tf.math

from hmsc.updaters.updateEta import _nngp_eta_prior_precision, updateEta

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
 
    if spatial_method=="Full":
      pass
    elif spatial_method=="NNGP":
      pass
    elif spatial_method=="GPP":  
      pass
    else:
      pass
    #aaa
    
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

    rLHyperparams = [None] * nr
    for r in range(nr):
        rLPar = {}
        rLPar["sDim"] = 0
        rLHyperparams[r] = rLPar

    return params, modelDims, modelData, priorHyperparams, rLHyperparams

@pytest.mark.parametrize("spatial_method", ["Full", "NNGP", "GPP", "None"])
def test_updateEta(spatial_method):

    params, modelDims, modelData, _, rLHyperparams = _simple_model(spatial_method)

    EtaTrue = params["Eta"]

    Eta = updateEta(params, modelDims, modelData, rLHyperparams)

    for r in range(modelDims["nr"]):
        assert_allclose(Eta[r], EtaTrue[r], atol=0.1)

    for r in range(modelDims["nr"]):
        assert_allclose(tf.reduce_mean(Eta[r]), tf.reduce_mean(EtaTrue[r]), atol=0.001)

def test_updateEta_shape():

    params, modelDims, modelData, _, rLHyperparams = _simple_model()
    
    EtaList = updateEta(params, modelDims, modelData, rLHyperparams)

    for r in range(modelDims["nr"]):
        assert tf.shape(EtaList[r])[0] == modelDims["np"][r]
        assert tf.shape(EtaList[r])[1] == modelDims["nf"][r]


def test_nngp_eta_prior_precision_uses_unit_major_factor_order():
    iW_factor_0 = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -0.5],
            [0.0, -0.5, 1.5],
        ]
    )
    iW_factor_1 = np.array(
        [
            [3.0, 0.25, 0.0],
            [0.25, 4.0, 0.75],
            [0.0, 0.75, 5.0],
        ]
    )

    prior = _nngp_eta_prior_precision(
        [iW_factor_0, iW_factor_1],
        Alpha=np.array([0, 1]),
        nu=3,
        nf=2,
    ).toarray()

    factor_0_positions = [0, 2, 4]
    factor_1_positions = [1, 3, 5]
    assert_allclose(prior[np.ix_(factor_0_positions, factor_0_positions)], iW_factor_0)
    assert_allclose(prior[np.ix_(factor_1_positions, factor_1_positions)], iW_factor_1)
    assert_allclose(prior[np.ix_(factor_0_positions, factor_1_positions)], 0.0)
    assert_allclose(prior[np.ix_(factor_1_positions, factor_0_positions)], 0.0)


def test_nngp_eta_prior_precision_rejects_alpha_factor_mismatch():
    with pytest.raises(ValueError, match="one spatial index per factor"):
        _nngp_eta_prior_precision([np.eye(2)], Alpha=np.array([0, 0]), nu=2, nf=1)
