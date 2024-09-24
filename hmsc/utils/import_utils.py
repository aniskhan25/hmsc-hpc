import numpy as np
import tensorflow as tf
from scipy import sparse

tfla, tfr, tfs, tfm = tf.linalg, tf.random, tf.sparse, tf.math


def load_model_data(hmscModel, importedInitParList, dtype=np.float64):

    Y = np.asarray(hmscModel.get("YScaled")).astype(dtype)
    T = np.asarray(hmscModel.get("TrScaled")).astype(dtype)
    C_import = hmscModel.get("C")
    if isinstance(hmscModel.get("XScaled"), dict):
        X = np.stack([np.asarray(hmscModel.get("XScaled")[x]) for x in hmscModel.get("XScaled")], 0)
    else:
        X = np.asarray(hmscModel.get("XScaled"))
    rhoGroup = np.asarray([0] * X.shape[-1])  # TODO replace once implemented in R as well
    # rhoGroup = np.asarray(hmscModel.get("rhoGroup")).astype(int) - 1
    Pi = np.asarray(hmscModel.get("Pi")).astype(int) - 1
    distr = np.asarray(hmscModel.get("distr")).astype(int)

    modelData = {}
    modelData["Y"] = Y
    modelData["X"] = X
    modelData["T"] = T
    if C_import is None or len(C_import) == 0:
        modelData["C"], modelData["eC"], modelData["VC"] = None, None, None
    else:
        C = np.asarray(C_import).astype(dtype)
        modelData["C"] = C
        modelData["eC"], modelData["VC"] = np.linalg.eigh(C)  # TODO replace once implemented in R as well
    modelData["rhoGroup"] = rhoGroup
    modelData["Pi"] = Pi
    modelData["distr"] = distr

    ny = int(hmscModel.get("ny")[0])
    ncsel = int(hmscModel.get("ncsel")[0])
    XSel = [{} for i in range(ncsel)]
    for i in range(ncsel):
        covGroup = np.array(hmscModel["XSelect"][i]["covGroup"]).astype(int) - 1
        spGroup = np.array(hmscModel["XSelect"][i]["spGroup"]).astype(int) - 1
        q = np.array(hmscModel["XSelect"][i]["q"]).astype(dtype)
        XSel[i]["covGroup"] = covGroup
        XSel[i]["spGroup"] = spGroup
        XSel[i]["q"] = q
    modelData["XSel"] = XSel

    # ncRRR = int(hmscModel.get("ncRRR")[0])
    if ("XRRRScaled" in hmscModel) and bool(hmscModel["XRRRScaled"]):
      XRRR = np.asarray(hmscModel["XRRRScaled"]).astype(dtype)
    else:
      XRRR = np.zeros([ny,0], dtype)
    modelData["XRRR"] = XRRR

    return modelData


def load_model_dims(hmscModel):

    ny = int(hmscModel.get("ny")[0])
    ns = int(hmscModel.get("ns")[0])
    nc = int(hmscModel.get("nc")[0])
    nt = int(hmscModel.get("nt")[0])
    nr = int(hmscModel.get("nr")[0])
    npVec = np.array(hmscModel.get("np"), int)
    ncsel = int(hmscModel.get("ncsel")[0])
    ncRRR = int(hmscModel.get("ncRRR")[0])
    ncNRRR = int(hmscModel.get("ncNRRR")[0])
    ncORRR = int(hmscModel.get("ncORRR")[0])
    nuRRR = int(hmscModel.get("nuRRR")[0])

    modelDims = {}
    modelDims["ny"] = ny
    modelDims["ns"] = ns
    modelDims["nc"] = nc
    modelDims["nt"] = nt
    modelDims["nr"] = nr
    modelDims["np"] = npVec
    modelDims["ncsel"] = ncsel
    modelDims["ncRRR"] = ncRRR
    modelDims["ncNRRR"] = ncNRRR
    modelDims["ncORRR"] = ncORRR
    modelDims["nuRRR"] = nuRRR

    return modelDims


def load_model_hyperparams(hmscModel, dataParList, dtype=np.float64):

    ns = int(np.squeeze(hmscModel.get("ns")))

    dataParams = {}
    if len(dataParList["Qg"]) == (ns * ns):  # TODO. need to review this condition
        dataParams["Qg"] = np.reshape(dataParList["Qg"], (ns, ns))
        dataParams["iQg"] = np.reshape(dataParList["iQg"], (ns, ns))
        dataParams["RQg"] = np.reshape(dataParList["RQg"], (ns, ns))
    else:
        dataParams["Qg"] = np.reshape(dataParList["Qg"], (101, ns, ns))
        dataParams["iQg"] = np.reshape(dataParList["iQg"], (101, ns, ns))
        dataParams["RQg"] = np.reshape(dataParList["RQg"], (101, ns, ns))

    return dataParams


def load_random_level_hyperparams(hmscModel, dataParList, dtype=np.float64):

    nr = int(np.squeeze(hmscModel.get("nr")))
    npVec = hmscModel.get("np")

    rLParams = [None] * nr
    for r in range(nr):
        rLName = list(hmscModel.get("rL").keys())[r]
        rLPar = {}
        rLPar["nu"] = dtype(hmscModel.get("rL")[rLName]["nu"][0])
        rLPar["a1"] = dtype(hmscModel.get("rL")[rLName]["a1"][0])
        rLPar["b1"] = dtype(hmscModel.get("rL")[rLName]["b1"][0])
        rLPar["a2"] = dtype(hmscModel.get("rL")[rLName]["a2"][0])
        rLPar["b2"] = dtype(hmscModel.get("rL")[rLName]["b2"][0])
        rLPar["nfMin"] = int(hmscModel.get("rL")[rLName]["nfMin"][0])
        rLPar["nfMax"] = int(hmscModel.get("rL")[rLName]["nfMax"][0])
        rLPar["sDim"] = int(hmscModel.get("rL")[rLName]["sDim"][0])
        rLPar["xDim"] = int(hmscModel.get("rL")[rLName]["xDim"][0])
        if rLPar["sDim"] > 0:
            rLPar["spatialMethod"] = hmscModel.get("rL")[rLName]["spatialMethod"][0]
            rLPar["alphapw"] = np.array(hmscModel.get("rL")[rLName]["alphapw"]).astype(dtype)
            gN = rLPar["alphapw"].shape[0]
            if rLPar["spatialMethod"] == "Full":
                distMat = np.reshape(dataParList["rLPar"][r]["distMat"], [npVec[r], npVec[r]]).astype(dtype)
                tmp = distMat / rLPar["alphapw"][:,0,None,None]
                tmp[np.isnan(tmp)] = 0
                rLPar["Wg"] = np.exp(-tmp)
                LWg = tfla.cholesky(rLPar["Wg"])
                rLPar["iWg"] = tfla.cholesky_solve(LWg, tf.eye(npVec[r], npVec[r], [gN], dtype))
                rLPar["LiWg"] = tfla.cholesky(rLPar["iWg"])
                rLPar["detWg"] = 2*tf.reduce_sum(tfm.log(tfla.diag_part(LWg)), -1)

            elif rLPar["spatialMethod"] == "GPP":
                nK = int(dataParList["rLPar"][r]["nKnots"][0])
                d12 = np.reshape(dataParList["rLPar"][r]["distMat12"], [npVec[r], nK]).astype(dtype)
                d22 = np.reshape(dataParList["rLPar"][r]["distMat22"], [nK, nK]).astype(dtype)
                W12 = d12 / rLPar["alphapw"][:,0,None,None]
                W12[np.isnan(W12)] = 0
                W12 = tf.exp(-W12)
                W22 = d22 / rLPar["alphapw"][:,0,None,None]
                W22[np.isnan(W22)] = 0
                W22 = tf.exp(-W22)
                LW22 = tfla.cholesky(W22)
                iW22 = tfla.cholesky_solve(LW22, tf.eye(nK, nK, [gN], dtype))
                dD = 1 - tf.einsum("gik,gkh,gih->gi", W12, iW22, W12)
                idD = dD**-1
                F = W22 + tf.einsum("gik,gi,gih->gkh", W12, idD, W12)
                LF = tfla.cholesky(F)
                iDW12 = tf.einsum("gi,gik->gik", idD, W12)
                detD = tf.reduce_sum(tfm.log(dD), -1) - 2*tf.reduce_sum(tfm.log(tfla.diag_part(LW22)), -1) + \
                  2*tf.reduce_sum(tfm.log(tfla.diag_part(LF)), -1)
                
                rLPar["nK"] = nK
                rLPar["idDg"] = idD
                rLPar["idDW12g"] = iDW12
                rLPar["Fg"] = F
                rLPar["iFg"] = tfla.cholesky_solve(tf.cast(tfla.cholesky(F), dtype=dtype), tf.eye(nK, nK, [gN], dtype))
                rLPar["detDg"] = detD
                
            elif rLPar["spatialMethod"] == "NNGP":
                indList, distList = dataParList["rLPar"][r]["indices"], dataParList["rLPar"][r]["distList"]
                iWList_csr = [None] * gN
                RiWList = [None] * gN
                detW = np.zeros([gN], dtype)
                indMat = np.concatenate([ind for ind in indList if (ind is not None and len(ind) > 0)], 1).T.astype(int) - 1
                for ag in range(gN):
                  alpha = rLPar["alphapw"][ag,0]
                  if alpha == 0:
                    RiWList[ag] = tfs.eye(npVec[r], dtype=dtype)
                    iWList_csr[ag] = sparse.eye(npVec[r], dtype=dtype)
                  else:
                    D = np.zeros([npVec[r]], dtype)
                    D[0] = 1
                    valList = [[]] * npVec[r]
                    for i in range(1,npVec[r]):
                      if indList[i] is not None and len(indList[i]) > 1:
                        Kp = np.exp(-np.array(distList[i]).astype(dtype)/alpha)
                        valList[i] = np.linalg.solve(Kp[:-1,:-1], Kp[:-1,-1])
                        D[i] = Kp[-1,-1] - np.matmul(Kp[:-1,-1], valList[i])
                      else:
                        D[i] = 1
                    iD05_csr = sparse.csr_array((D**-0.5, (np.arange(npVec[r]),np.arange(npVec[r]))), [npVec[r]]*2, dtype=dtype)
                    A = sparse.csr_array((np.concatenate(valList), (indMat[:,0],indMat[:,1])), [npVec[r]]*2, dtype=dtype)
                    B = sparse.eye(npVec[r], dtype=dtype) - A
                    RiW = iD05_csr @ B
                    iWList_csr[ag] = RiW.T @ RiW
                    RiWList[ag] = tfs.reorder(tfs.SparseTensor(np.stack(RiW.nonzero(), 1), RiW[RiW.nonzero()], [npVec[r]]*2))
                    detW[ag] = np.sum(np.log(D))
                    
                rLPar["iWList_csr"] = iWList_csr
                rLPar["RiWList"] = RiWList
                rLPar["detWg"] = detW
        
        if rLPar["xDim"] > 0:
            rLPar["xMat"] = np.array(hmscModel.get("rL")[rLName]["xMat"]) # TODO. unsure about dtype
        
        rLParams[r] = rLPar

    return rLParams


def load_prior_hyperparams(hmscModel, dtype=np.float64):

    mGamma = np.asarray(hmscModel.get("mGamma")).astype(dtype)
    UGamma = np.asarray(hmscModel.get("UGamma")).astype(dtype)
    f0 = np.squeeze(hmscModel.get("f0")).astype(dtype)
    V0 = np.squeeze(hmscModel.get("V0")).astype(dtype)
    rhopw = np.asarray(hmscModel.get("rhopw")).astype(dtype)
    aSigma = np.asarray(hmscModel.get("aSigma")).astype(dtype)
    bSigma = np.asarray(hmscModel.get("bSigma")).astype(dtype)
    nuRRR = np.squeeze(hmscModel.get("nuRRR")).astype(dtype)
    a1RRR = np.squeeze(hmscModel.get("a1RRR")).astype(dtype)
    b1RRR = np.squeeze(hmscModel.get("b1RRR")).astype(dtype)
    a2RRR = np.squeeze(hmscModel.get("a2RRR")).astype(dtype)
    b2RRR = np.squeeze(hmscModel.get("b2RRR")).astype(dtype)

    priorHyperParams = {}
    priorHyperParams["mGamma"] = mGamma
    priorHyperParams["UGamma"] = UGamma
    priorHyperParams["iUGamma"] = tfla.inv(UGamma).numpy()
    priorHyperParams["f0"] = f0
    priorHyperParams["V0"] = V0
    priorHyperParams["rhopw"] = rhopw
    priorHyperParams["aSigma"] = aSigma
    priorHyperParams["bSigma"] = bSigma
    priorHyperParams["nuRRR"] = nuRRR
    priorHyperParams["a1RRR"] = a1RRR
    priorHyperParams["b1RRR"] = b1RRR
    priorHyperParams["a2RRR"] = a2RRR
    priorHyperParams["b2RRR"] = b2RRR

    return priorHyperParams


def init_params(importedInitParList, modelData, modelDims, rLHyperparams, dtype=np.float64):

    initParList = [None] * len(importedInitParList)
    for chainInd, importedInitPar in enumerate(importedInitParList):
        # Z = tf.constant(importedInitPar["Z"], dtype=dtype)
        Beta = tf.constant(importedInitPar["Beta"], dtype=dtype)
        Gamma = tf.constant(importedInitPar["Gamma"], dtype=dtype)
        iV = tfla.inv(tf.constant(importedInitPar["V"], dtype=dtype))
        rhoInd = (tf.cast(tf.constant(importedInitPar["rho"]), tf.int32) - 1)  # TODO replace once implemented in R as well
        sigma = tf.constant(importedInitPar["sigma"], dtype=dtype)
        EtaList = [tf.constant(Eta, dtype=dtype) for Eta in importedInitPar["Eta"]]
        AlphaIndList = [tf.cast(tf.constant(AlphaInd), tf.int32) - 1 for AlphaInd in importedInitPar["Alpha"]]
        LambdaList, PsiList, DeltaList  = [None] * modelDims["nr"], [None] * modelDims["nr"], [None] * modelDims["nr"]
        for r, (Lambda, Psi, Delta, Eta, rLPar) in enumerate(zip(importedInitPar["Lambda"], importedInitPar["Psi"], importedInitPar["Delta"], 
                                                                 EtaList, rLHyperparams)):
            nf = Eta.shape[1]
            DeltaList[r] = tf.constant(Delta, dtype=dtype)  
            if rLPar["xDim"] == 0:
                LambdaList[r] = tf.constant(Lambda, dtype=dtype)
                PsiList[r] = tf.constant(Psi, dtype=dtype)
            else:
                LambdaList[r] = tf.transpose(tf.reshape(tf.constant(Lambda, dtype=dtype), [rLPar["xDim"],modelDims["ns"],nf]), [2,1,0])
                PsiList[r] = tf.transpose(tf.reshape(tf.constant(Psi, dtype=dtype), [rLPar["xDim"],modelDims["ns"],nf]), [2,1,0])

        BetaSel = [tf.constant(BetaSel, dtype=tf.bool) for BetaSel in importedInitPar["BetaSel"]]

        if modelDims["ncsel"] > 0:
          ns, nc = modelDims["ns"], modelDims["nc"]
          X, XSel = modelData["X"], modelData["XSel"]
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
          Xeff = tf.constant(modelData["X"], dtype=dtype)

        XRRR = modelData["XRRR"]
        PsiRRR = tf.constant(importedInitPar["PsiRRR"], dtype) if "PsiRRR" in importedInitPar else tf.zeros([0,0], dtype)
        DeltaRRR = tf.squeeze(tf.constant(importedInitPar["DeltaRRR"], dtype), 1) if "DeltaRRR" in importedInitPar else tf.zeros([0], dtype)
        wRRR = tf.constant(importedInitPar["wRRR"], dtype) if "wRRR" in importedInitPar else tf.zeros([0,0], dtype)
        XeffRRR = tf.einsum("ik,hk->ih", XRRR, wRRR)
        if Xeff.ndim == 2:
          Xeff = tf.concat([Xeff, XeffRRR], axis=-1)
        else:
          Xeff = tf.concat([Xeff, tf.repeat(tf.expand_dims(XeffRRR,0), modelDims["ns"], 0)], axis=-1)

        initPar = {}
        initPar["Z"] = None
        initPar["Beta"] = Beta
        initPar["Gamma"] = Gamma
        initPar["iV"] = iV
        initPar["rhoInd"] = rhoInd
        initPar["sigma"] = sigma
        initPar["Lambda"] = LambdaList
        initPar["Psi"] = PsiList
        initPar["Delta"] = DeltaList
        initPar["Eta"] = EtaList
        initPar["AlphaInd"] = AlphaIndList
        initPar["BetaSel"] = BetaSel
        initPar["PsiRRR"] = PsiRRR
        initPar["DeltaRRR"] = DeltaRRR
        initPar["wRRR"] = wRRR
        initPar["Xeff"] = Xeff
        initParList[chainInd] = initPar

    return initParList
