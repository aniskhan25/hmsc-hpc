import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfla, tfr, tfs = tf.math, tf.linalg, tf.random, tf.sparse

@tf_named_func("betaLambda")
def updateBetaLambda(params, data, priorHyperparams, rLHyperparams=None, dtype=np.float64):
    """Update conditional updater(s):
    Beta - species niches, and
    Lambda - species loadings.

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s).
            Z - latent variables
            Gamma - influence of traits on species niches
            iV - inverse residual covariance of species niches
            Eta - site loadings
            Psi - local shrinage species loadings (lambda's prior)
            Delta - delta global shrinage species loadings (lambda's prior)
            sigma - residual variance
            Y - community matrix
            X - environmental data
            T - species trait data
            Pi - study design
    """

    Z = params["Z"]
    iD = params["iD"]
    Gamma = params["Gamma"]
    iV = params["iV"]
    rhoInd = params["rhoInd"]
    EtaList = params["Eta"]
    PsiList = params["Psi"]
    DeltaList = params["Delta"]
    X = params["Xeff"]
    if rLHyperparams is None:
        rLHyperparams = [{"xDim": 0} for _ in EtaList]

    Yo = data["Yo"]
    distr = data["distr"]
    T = data["T"]
    C, eC, VC = data["C"], data["eC"], data["VC"]
    rhoGroup = data["rhoGroup"]
    Pi = data["Pi"]
    rhopw = priorHyperparams["rhopw"]

    ny, nc = X.shape[-2:]
    _, ns = Z.shape
    nr = len(EtaList)
    nfVec = tf.stack([tf.shape(Eta)[-1] for Eta in EtaList])
    nLambdaVec = tf.stack(
        [
            tf.shape(Eta)[-1] * tf.cast(rLPar.get("xDim", 0) if rLPar.get("xDim", 0) > 0 else 1, tf.int32)
            for Eta, rLPar in zip(EtaList, rLHyperparams)
        ]
    ) if nr > 0 else tf.zeros([0], dtype=tf.int32)
    nLambdaSum = tf.cast(tf.reduce_sum(nLambdaVec), tf.int32)
    na = nc + nLambdaSum

    PiEtaList = [None] * nr
    for r, Eta in enumerate(EtaList):
      PiEtaList[r] = _random_level_predictors(Eta, rLHyperparams[r], Pi[:, r])
    
    # print(["rank", X.shape, len(X.shape.as_list()), tf.rank(X)])
    if len(X.shape.as_list()) == 2:
      # tf.print(tf.rank(X))
      XE = tf.concat([X] + PiEtaList, axis=-1)
    else:
      if nr == 0:
        XE = X
      else:
        XE = tf.concat([X, tf.repeat(tf.expand_dims(tf.concat(PiEtaList, axis=-1), 0), ns, 0)], axis=-1)
    

    GammaT = tf.matmul(Gamma, T, transpose_b=True)
    Mu = tf.concat([GammaT, tf.zeros([nLambdaSum, ns], dtype)], axis=0)
    if nr > 0:
      LambdaPriorPrec = tf.concat(
          [_flatten_random_slope_loadings(Psi) * tf.repeat(tfm.cumprod(Delta, -2), _lambda_x_dim(Psi), axis=-2)
           for Psi, Delta in zip(PsiList, DeltaList)],
          axis=-2,
      )
      
    if C is None:
      if nr > 0:
        iK11_op = tfla.LinearOperatorFullMatrix(iV)
        iK22_op = tfla.LinearOperatorDiag(tf.transpose(LambdaPriorPrec))
        iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
      else:
        iK = iV

      if np.all(Yo == True) and np.sum(distr[:,0] == 2) == ns: #special case with all iD[i,j]==1
        iDZ = Z
        iU = iK + tf.matmul(XE, XE, transpose_a=True, name="iU.2")
      else:
        iDZ = iD * Z
        if tf.reduce_all(iD == iD[0,:]):
          iU = iK + iD[0,:,None,None] * tf.matmul(XE, XE, transpose_a=True, name="iU.2")
        else:
          iD05_XE = tf.multiply(tfla.matrix_transpose(iD)[:,:,None]**0.5, XE, name="iD05_XE")
          iU = iK + tf.matmul(iD05_XE, iD05_XE, transpose_a=True, name="iU.2")
          
      A1 = tf.matmul(iK, tfla.matrix_transpose(Mu)[:,:,None], name="A.1")
      if len(XE.shape.as_list()) == 2:
        A2 = tf.matmul(iDZ, XE, transpose_a=True, name="A.2")[:,:,None]
      else:
        A2 = tf.einsum("jik,ij->jk", XE, iDZ, name="A.2")[:,:,None]
      A = A1 + A2

      LiU = tfla.cholesky(iU, name="LiU")
      M = tfla.cholesky_solve(LiU, A, name="M")
      BetaLambda = tf.transpose(tf.squeeze(M + tfla.triangular_solve(LiU, tfr.normal([ns,na,1], dtype=dtype, name="BetaLambda.2"), adjoint=True), -1))
      BetaLambda = BetaLambda
    else:
      rhoVec = tf.gather(rhopw[:,0], tf.gather(rhoInd, rhoGroup))
      eQ = rhoVec[:,None]*eC + (1-rhoVec)[:,None]
      eiQ05 = tfm.rsqrt(eQ)
      eiQ_block = tf.expand_dims(eiQ05, 0) * tf.expand_dims(eiQ05, 1)
      PB_stack = tf.einsum("ij,ckj,gj->cikg", VC, eiQ_block*iV[:,:,None], VC, name="PB_stack")
      PB = tf.reshape(PB_stack, [nc*ns,nc*ns])
      iK11_op = tfla.LinearOperatorFullMatrix(PB)
      if nr > 0:
        iK22_op = tfla.LinearOperatorDiag(tf.reshape(LambdaPriorPrec, [nfSum*ns]))
        iK = tfla.LinearOperatorBlockDiag([iK11_op, iK22_op]).to_dense()
      else:
        iK = iK11_op.to_dense()
              
      if len(XE.shape.as_list()) == 2:
        XE_iD_XET = tf.einsum("ic,ij,ik->ckj", XE, iD, XE, name="XE_iD_XET")
        m0 = tf.matmul(iK, tf.reshape(Mu, [na*ns,1])) + tf.reshape(tf.matmul(XE, iD*Z, transpose_a=True), [na*ns,1])
      else:
        XE_iD_XET = tf.einsum("jic,ij,jik->ckj", XE, iD, XE, name="XE_iD_XET")
        m0 = tf.matmul(iK, tf.reshape(Mu, [na*ns,1])) + tf.reshape(tf.einsum("jik,ij->kj", XE, iD*Z, name="m0.2"), [na*ns,1])
      
      iU = iK + tf.reshape(tf.transpose(tfla.diag(XE_iD_XET), [0,2,1,3]), [na*ns]*2)
      LiU = tfla.cholesky(iU)        
      m = tfla.cholesky_solve(LiU, m0)
      BetaLambda = tf.reshape(m + tfla.triangular_solve(LiU, tfr.normal(shape=[na*ns,1], dtype=dtype), adjoint=True), [na,ns])
    
    if nr > 0:
      BetaLambdaList = tf.split(BetaLambda, tf.concat([tf.constant([nc], tf.int32), nLambdaVec], -1), axis=-2)
      BetaNew = tf.ensure_shape(BetaLambdaList[0], [nc,ns])
      LambdaListNew = [
          _restore_random_slope_loadings(LambdaFlat, LambdaOld, rLPar)
          for LambdaFlat, LambdaOld, rLPar in zip(BetaLambdaList[1:], params["Lambda"], rLHyperparams)
      ]
    else:
      BetaNew, LambdaListNew = tf.ensure_shape(BetaLambda, [nc,ns]), []

    # tf.print(BetaNew)
    return BetaNew, LambdaListNew


def _random_level_predictors(Eta, rLPar, Pi):
    xMat = rLPar.get("xMat")
    PiEta = tf.gather(Eta, Pi)
    if xMat is None:
        return PiEta
    PiX = tf.gather(tf.convert_to_tensor(xMat, dtype=Eta.dtype), Pi)
    return tf.reshape(tf.einsum("ih,ik->ihk", PiEta, PiX), [tf.shape(PiEta)[0], -1])


def _flatten_random_slope_loadings(values):
    if len(values.shape.as_list()) == 2:
        return values
    return tf.reshape(tf.transpose(values, [0, 2, 1]), [tf.shape(values)[0] * tf.shape(values)[2], tf.shape(values)[1]])


def _restore_random_slope_loadings(values, like, rLPar):
    if int(rLPar.get("xDim", 0)) == 0:
        return tf.ensure_shape(values, [None, like.shape[-1]])
    nf = tf.shape(like)[0]
    ns = tf.shape(like)[1]
    x_dim = tf.shape(like)[2]
    restored = tf.transpose(tf.reshape(values, [nf, x_dim, ns]), [0, 2, 1])
    return tf.ensure_shape(restored, [None, like.shape[1], like.shape[2]])


def _lambda_x_dim(values):
    if len(values.shape.as_list()) == 2:
        return 1
    return tf.shape(values)[2]
