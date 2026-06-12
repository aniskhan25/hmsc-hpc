import numpy as np
import tensorflow as tf
from hmsc.utils.tf_named_func import tf_named_func
tfm, tfr = tf.math, tf.random

@tf_named_func("lambdaPriors")
def updateLambdaPriors(params, rLHyperparams, dtype=np.float64):
    """Update prior(s) for each random level:
    Psi - local shrinage species loadings (lambda's prior), and
    Delta - delta global shrinage species loadings (lambda's prior).

    Parameters
    ----------
    params : dict
        The initial value of the model parameter(s):
        Lambda - species loadings
        Delta - delta global shrinage species loadings (lambda's prior)
        nu -
        a1 -
        b1 -
        a2 -
        b2 -
    """

    LambdaList = params["Lambda"]
    DeltaList = params["Delta"]

    nr = len(LambdaList)
    PsiNew, DeltaNew = [None] * nr, [None] * nr
    for r, (Lambda, Delta, rLPar) in enumerate(zip(LambdaList, DeltaList, rLHyperparams)):

        nu = rLPar["nu"]
        a1 = rLPar["a1"]
        b1 = rLPar["b1"]
        a2 = rLPar["a2"]
        b2 = rLPar["b2"]

        ns = Lambda.shape[1] if len(Lambda.shape.as_list()) == 3 else Lambda.shape[-1]
        n_loading_columns = tf.reduce_prod(tf.shape(Lambda)[1:])
        nf = tf.shape(Lambda)[0]
        if nf > 0:
            aDelta = tf.concat([a1 * tf.ones([1, 1], dtype), a2 * tf.ones([nf-1, 1], dtype)], 0)
            bDelta = tf.concat([b1 * tf.ones([1, 1], dtype), b2 * tf.ones([nf-1, 1], dtype)], 0)
            Lambda2 = Lambda**2
            Tau = tfm.cumprod(Delta, 0)
            aPsi = nu/2 + 0.5
            TauForLambda = tf.reshape(Tau, [nf] + [1] * (len(Lambda.shape.as_list()) - 1))
            bPsi = nu/2 + 0.5 * Lambda2 * TauForLambda
            PsiNew[r] = tf.squeeze(tfr.gamma([1], aPsi, bPsi, dtype=dtype), 0)
            M = PsiNew[r] * Lambda2
            rowSumM = tf.reduce_sum(M, axis=tf.range(1, tf.rank(M)))
            for h in tf.range(nf):
                Tau = tfm.cumprod(Delta, 0)
                ad = aDelta[h, :] + 0.5 * tf.cast(n_loading_columns, dtype) * tf.cast(nf-h, dtype)
                bd = bDelta[h, :] + 0.5 * tf.reduce_sum(Tau[h:, :] * rowSumM[h:, None], 0) / Delta[h, :]
                Delta = tf.tensor_scatter_nd_update(Delta, [[h]], tfr.gamma([1], ad, bd, dtype=dtype))
            DeltaNew[r] = Delta
            if len(Lambda.shape.as_list()) == 3:
                PsiNew[r] = tf.ensure_shape(PsiNew[r], [None, ns, Lambda.shape[2]])
            else:
                PsiNew[r] = tf.ensure_shape(PsiNew[r], [None, ns])
        else:
            if len(Lambda.shape.as_list()) == 3:
                PsiNew[r] = tf.zeros([0, ns, Lambda.shape[2]], dtype)
            else:
                PsiNew[r] = tf.zeros([0, ns], dtype)
            DeltaNew[r] = tf.zeros([0, 1], dtype)
    return PsiNew, DeltaNew
