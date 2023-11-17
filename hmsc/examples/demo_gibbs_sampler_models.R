# library(devtools)
# install_github("hmsc-r/HMSC")
library(Hmsc)
library(jsonify)
library(vioplot)
library(abind)
RS = 1
set.seed(RS)
nParallel = 8
flagFitR = 1

path = getwd()

#### Step 1. Load model ####

load(file = file.path(path, "examples/data", "unfitted_models_2.RData"))
models
experiments = list(
  M1=list(name="model1",id=1),
  M2=list(name="model2",id=2),
  M3=list(name="model3",id=3),
  M4=list(name="model4",id=4),
  M5=list(name="model5",id=5),
  M6=list(name="model6",id=6),
  M7=list(name="model7",id=7),
  M8=list(name="model8",id=8),
  M9=list(name="model9",id=9),
  M10=list(name="model10",id=10),
  M11=list(name="model11",id=NA)
)

selected_experiment = experiments$M4
if(!is.na(selected_experiment$id)){
  m = models[[selected_experiment$id]]
}

if (selected_experiment$name == experiments$M1$name) {
  nChains = 8
  nSamples = 250
  thin = 10
} else if (selected_experiment$name == experiments$M2$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  rLSite = m$ranLevels$site
  rLTime = m$ranLevels$time
  rLSiteNew = HmscRandomLevel(units=rLSite$pi)
  rLTimeNew = HmscRandomLevel(units=rLTime$pi)
  # rLSiteNew = m$ranLevels$site
  # rLTimeNew = m$ranLevels$time
  # YNew = matrix(colMeans(m$Y, na.rm=TRUE), nrow(m$Y), ncol(m$Y), byrow=TRUE)
  # YNew[!is.na(m$Y)] = m$Y[!is.na(m$Y)]
  YNew = m$Y
  XFormulaNew = as.formula("~poly(temp_avg, degree = 2, raw = TRUE) + season")
  mNew = Hmsc(Y=YNew, XFormula=XFormulaNew, XData=m$XData, TrFormula=m$TrFormula, TrData=m$TrData,
              distr=m$distr, studyDesign=m$studyDesign, ranLevels=list(site=rLSiteNew,time=rLTimeNew)) #
  m = mNew
} else if (selected_experiment$name == experiments$M3$name) {
  nChains = 8
  nSamples = 250
  thin = 1
} else if (selected_experiment$name == experiments$M4$name) { ################################
  nChains =  8
  nSamples = 250
  thin = 128
  
  ny = min(500, m$ny)
  nc = min(10, m$nc)
  nt = min(1, m$nt)
  nf = 5
  
  X = m$X[1:ny,1:nc,drop=FALSE]
  Tr = m$Tr[1:ns,1:nt,drop=FALSE]
  studyDesign = m$studyDesign[1:ny,]
  for(h in 1:ncol(studyDesign)) studyDesign[,h] = factor(studyDesign[,h], levels=unique(studyDesign[,h]))
  
  # Gamma = c(0.3,-1,0.1,1,-1)[1:nc]
  # Beta = 0.2*matrix(rnorm(nc*m$ns), nc, m$ns) + Gamma
  # Eta = matrix(rnorm(ny*nf), ny, nf)
  # Lambda = matrix(rnorm(nf*m$ns), c(nf,m$ns))
  # sigma = rep(1, m$ns)
  # Y = (X %*% Beta + Eta %*% Lambda + matrix(sigma,ny,m$ns,byrow=TRUE)*matrix(rnorm(ny*m$ns),ny,m$ns) > 0) + 0 #
  Y = m$Y[1:ny,]
  rlUnit = m$ranLevels$unit
  rlUnit = setPriors(rlUnit, nfMin=1, nfMax=ns)
  m = Hmsc(Y=Y, X=X, XScale=TRUE, Tr=Tr, TrScale=FALSE, distr="probit", studyDesign=studyDesign, ranLevels=list(unit=rlUnit)) #unit=m$ranLevels$unit
} else if (selected_experiment$name == experiments$M5$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
  transient = nSamples*thin
} else if (selected_experiment$name == experiments$M6$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  #m$X = 0.5*(m$X[[1]] + m$X[[2]])
  #m$XScaled = 0.5*(m$XScaled[[1]] + m$XScaled[[2]])
  #m$XData = 0.5*(m$XData[[1]] + m$XData[[2]])
} else if (selected_experiment$name == experiments$M7$name) {
  print("Not Implemented: XScaled not found error for engine=pass")
  stop("Not covered")
} else if (selected_experiment$name == experiments$M8$name) {
  nChains = 8
  nSamples = 250
  thin = 1000
  set.seed(1)
  indSub = sort(sample(nrow(m$studyDesign), 200))
  studyDesignNew = m$studyDesign[indSub,]
  for(r in 1:ncol(studyDesignNew)) studyDesignNew[,r] = factor(studyDesignNew$year)
  rLYearNew = HmscRandomLevel(units=levels(studyDesignNew$year))
  rLSiteNew = HmscRandomLevel(units=levels(studyDesignNew$site.code))
  mNew = Hmsc(Y=m$Y[indSub,], XFormula=m$XFormula, XData=m$XData[indSub,], XRRRFormula=m$XRRRFormula, XRRRData=m$XRRRData[indSub,],
              distr=m$distr, studyDesign=studyDesignNew) #, ranLevels=list(year=rLYearNew,site.code=rLSiteNew))
  m = mNew
} else if (selected_experiment$name == experiments$M9$name) {
  nChains = 8
  nSamples = 250
  thin = 10
  set.seed(1)
  indSub = sort(sample(nrow(m$studyDesign), 200))
  studyDesignNew = m$studyDesign[indSub,]
  for(r in 1:ncol(studyDesignNew)) studyDesignNew[,r] = factor(studyDesignNew$year)
  mNew = Hmsc(Y=m$Y[indSub,], XFormula=m$XFormula, XData=m$XData[indSub,], XSelect=m$XSelect,
              distr=m$distr, studyDesign=studyDesignNew)
  m = mNew
} else if (selected_experiment$name == experiments$M10$name) {
  nChains = 8
  nSamples = 250
  thin = 1000
  # mNew = Hmsc(Y=m$Y, XFormula=m$XFormula, XData=m$XData, distr=m$distr, studyDesign=m$studyDesign, ranLevels=list())
  # mNew = setPriors(mNew, V0=10^2*diag(m$nc), UGamma=10^2*diag(m$nc))
  # m = mNew
} else if (selected_experiment$name == experiments$M11$name) {
  nc = 5
  ncr = 2
  ns = 53
  ny = 213
  nf = 3
  X = cbind(rep(1,ny), as.matrix(matrix(rnorm(ny*(nc-1)), ny, nc-1)))
  colnames(X) = c("intercept", sprintf("c%d", 1:(nc-1)))
  Beta = matrix(rnorm(nc*ns), nc, ns)
  LFix = X %*% Beta
  LRan = NA * LFix
  Eta = matrix(rnorm(ny*nf), ny, nf)
  Lambda = array(rnorm(nf*ns*ncr), c(nf,ns,ncr))
  XRan = cbind(rep(1,ny), as.matrix(matrix(rnorm(ny*(ncr-1)), ny, ncr-1)))
  rownames(XRan) = sprintf("su%.4d", 1:ny)
  studyDesign = data.frame(su=as.factor(rownames(XRan)))
  for(i in 1:ny){
    LambdaLocal = Lambda[,,1]*XRan[i,1]
    for(k in 1+seq(ncr-1)) LambdaLocal = Lambda[,,k]*XRan[i,k]
    LRan[i,] = Eta[i,] %*% LambdaLocal
  }
  sigma = rep(1, ns)
  Y = LFix + LRan + matrix(sigma,ny,ns,byrow=TRUE)*matrix(rnorm(ny*ns),ny,ns)
  rLSu = HmscRandomLevel(xData=XRan)
  m = Hmsc(Y=Y, X=X, distr="normal", studyDesign=studyDesign, ranLevels=list(su=rLSu))
  
  nChains = 8
  nSamples = 250
  thin = 1
} else{
  stop("Not covered")
}
transient = nSamples*thin
verbose = thin*1

#### Step 2. Export initial model ####

set.seed(RS+42)
init_obj = sampleMcmc(m, samples=nSamples, thin=thin,
                      transient=transient,
                      nChains=nChains, verbose=verbose, engine="pass")

init_file_name = sprintf("TF-init-obj-%s.rds", selected_experiment$name)
init_file_path = file.path(path, "examples/data", init_file_name)
fitR_file_name = sprintf("R-fit-%s_thin%.4d.RData", selected_experiment$name, thin)
fitR_file_path = file.path(path, "examples/data", fitR_file_name)
python_file_name = "run_gibbs_sampler.py"
python_file_path = file.path(path, "examples", python_file_name)
postList_file_name = sprintf("TF-postList-obj-%s.rds", selected_experiment$name)
# postList_file_name = sprintf("TF-postList-obj-%s_tf_thin%.4d.rds", selected_experiment$name, thin)
postList_file_path = file.path(path, "examples/data", postList_file_name)

nr = init_obj[["hM"]][["nr"]]
rLNames = init_obj[["hM"]][["ranLevelsUsed"]]
for (r in seq_len(nr)) {
  rLName = rLNames[[r]]
  init_obj[["hM"]][["rL"]][[rLName]][["s"]] = NULL
  init_obj[["hM"]][["ranLevels"]][[rLName]][["s"]] = NULL
  spatialMethod = init_obj[["hM"]][["rL"]][[r]][["spatialMethod"]]
  if (!is.null(spatialMethod)) {
    if (spatialMethod == "NNGP") {
      gN = length(init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]])
  
      for (i in seq_len(gN)) {
        iWg = as(init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]][[i]], "dgTMatrix")
        RiWg = as(init_obj[["dataParList"]][["rLPar"]][[r]][["RiWg"]][[i]], "dgTMatrix")
        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgi"]][[i]] = iWg@i
        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgj"]][[i]] = iWg@j
        init_obj[["dataParList"]][["rLPar"]][[r]][["iWgx"]][[i]] = iWg@x
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgi"]][[i]] = RiWg@i
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgj"]][[i]] = RiWg@j
        init_obj[["dataParList"]][["rLPar"]][[r]][["RiWgx"]][[i]] = RiWg@x
      }
      init_obj[["dataParList"]][["rLPar"]][[r]][["iWg"]] = NULL
      init_obj[["dataParList"]][["rLPar"]][[r]][["RiWg"]] = NULL
    }
    else if (spatialMethod == "GPP") {
      init_obj[["dataParList"]][["rLPar"]][[r]][["nK"]] = nrow(init_obj[["dataParList"]][["rLPar"]][[1]][["Fg"]])
    }
  }
}

saveRDS(to_json(init_obj), file = init_file_path, compress=TRUE)

#### Step 3. Run R code ####

#
# Generate sampled posteriors in R
#

if(flagFitR){
  set.seed(RS+42)
  startTime = proc.time()
  obj.R = sampleMcmc(m, samples = nSamples, thin = thin,
                     transient = transient, 
                     nChains = nChains, nParallel=min(nChains,nParallel),
                     verbose = verbose, updater=list(Gamma2=FALSE, GammaEta=FALSE)) #fitted by R
  elapsedTime = proc.time() - startTime
  print(elapsedTime)
  save(obj.R, elapsedTime, file=fitR_file_path)
} else{
  load(file=fitR_file_path)
}

#### Step 4. Run TF code ####

python_cmd = paste("python", sprintf("'%s'",python_file_path), 
                   "--samples", nSamples,
                   "--transient", transient,
                   "--thin", thin,
                   "--verbose", verbose,
                   "--input", init_file_name, 
                   "--output", postList_file_name,
                   "--path", sprintf("'%s'",path))
print(python_cmd)
aaa
#
# Set RStudio to TF env
#
# my_conda_env_name = "tensorflow" # name of my conda TF env

# Start one-time python setup
# INFO. one-time steps to set python for RStudio/reticulate
# INFO. requires a session restart for Sys.setenv() to take effect

# library(tidyverse)
# py_bin <- reticulate::conda_list() %>%
#   filter(name == my_conda_env_name) %>%
#   pull(python)
#
# Sys.setenv(RETICULATE_PYTHON = py_bin)

# End one-time python setup

# library(reticulate)
# use_condaenv(my_conda_env_name, required=TRUE) # activate the TF env
# repl_python()

#
# Generate sampled posteriors in TF
#
system(paste("chmod a+x", sprintf("'%s'",python_file_path))) # set file permissions for shell execution
system("python --version", wait=TRUE)
system(python_cmd, wait=TRUE) # run TF gibbs sampler




#### Step 5. Import TF posteriors ####################################################################################

postList.TF <- from_json(readRDS(file = postList_file_path)[[1]])

# postList_file_str <- paste(readLines(postList_file_path), collapse="\n")
# s = '{"0":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"1":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"2":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"3":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"4":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"5":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"6":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}},"7":{"0":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null},"1":{"wRRR":null,"rho":null,"PsiRRR":null,"DeltaRRR":null}}}'
# from_json(s)

names(postList.TF) = NULL
for (chain in seq_len(nChains)) {
  names(postList.TF[[chain]]) = NULL
}

obj.TF = init_obj$hM
obj.TF[["postList"]] = postList.TF

# tmp = obj.R$postList[[1]]
# obj.R$postList[[1]] = obj.R$postList[[2]]
# tmp = obj.TF$postList[[1]]
# obj.TF$postList[[1]] = obj.TF$postList[[2]]

obj.TF$samples = nSamples
obj.TF$thin = thin
obj.TF$transient = transient

#
# Rescaling Beta/Gamma; copied from combineParameters.R; need to revisit this section
#
nt = obj.TF[["nt"]]
TrInterceptInd = obj.TF[["TrInterceptInd"]]
TrScalePar = obj.TF[["TrScalePar"]]
ncNRRR = obj.TF[["ncNRRR"]]
ncRRR = obj.TF[["ncRRR"]]
ncsel = obj.TF[["ncsel"]]
XScalePar = obj.TF[["XScalePar"]]
XInterceptInd = obj.TF[["XInterceptInd"]]
XRRRScalePar = obj.TF[["XRRRScalePar"]]

for (chain in seq_len(nChains)) {
  for (sample in seq_len(nSamples)) {
    Beta = obj.TF[["postList"]][[chain]][[sample]][["Beta"]]
    BetaSel = obj.TF[["postList"]][[chain]][[sample]][["BetaSel"]]
    if(is.matrix(BetaSel)){
      BetaSel = split(BetaSel, rep(1:nrow(BetaSel), ncol(BetaSel)))
    }
    Gamma = obj.TF[["postList"]][[chain]][[sample]][["Gamma"]]
    iV = obj.TF[["postList"]][[chain]][[sample]][["iV"]]
    rho = obj.TF$rhopw[obj.TF[["postList"]][[chain]][[sample]][["rhoInd"]], 1]
    sigma = obj.TF[["postList"]][[chain]][[sample]][["sigma"]]

    for(p in 1:nt){
      me = TrScalePar[1,p]
      sc = TrScalePar[2,p]
      if(me!=0 || sc!=1){
        Gamma[,p] = Gamma[,p]/sc
        if(!is.null(TrInterceptInd)){
          Gamma[,TrInterceptInd] = Gamma[,TrInterceptInd] - me*Gamma[,p]
        }
      }
    }

    for(k in 1:ncNRRR){
      me = XScalePar[1,k]
      sc = XScalePar[2,k]
      if(me!=0 || sc!=1){
        Beta[k,] = Beta[k,]/sc
        Gamma[k,] = Gamma[k,]/sc
        if(!is.null(XInterceptInd)){
          Beta[XInterceptInd,] = Beta[XInterceptInd,] - me*Beta[k,]
          Gamma[XInterceptInd,] = Gamma[XInterceptInd,] - me*Gamma[k,]
        }
        iV[k,] = iV[k,]*sc
        iV[,k] = iV[,k]*sc
      }
    }

    for(k in seq_len(ncRRR)){
      me = XRRRScalePar[1,k]
      sc = XRRRScalePar[2,k]
      if(me!=0 || sc!=1){
        Beta[ncNRRR+k,] = Beta[ncNRRR+k,]/sc
        Gamma[ncNRRR+k,] = Gamma[ncNRRR+k,]/sc
        if(!is.null(XInterceptInd)){
          Beta[XInterceptInd,] = Beta[XInterceptInd,] - me*Beta[ncNRRR+k,]
          Gamma[XInterceptInd,] = Gamma[XInterceptInd,] - me*Gamma[ncNRRR+k,]
        }
        iV[ncNRRR+k,] = iV[ncNRRR+k,]*sc
        iV[,ncNRRR+k] = iV[,ncNRRR+k]*sc
      }
    }

    for (i in seq_len(ncsel)){
      XSel = obj.TF$XSelect[[i]]
      for (spg in 1:length(XSel$q)){
        if(!BetaSel[[i]][spg]){
          fsp = which(XSel$spGroup==spg)
          Beta[XSel$covGroup,fsp]=0
        }
      }
    }

    obj.TF[["postList"]][[chain]][[sample]][["Beta"]] = Beta
    obj.TF[["postList"]][[chain]][[sample]][["Gamma"]] = Gamma
    obj.TF[["postList"]][[chain]][[sample]][["V"]] = chol2inv(chol(iV))
    obj.TF[["postList"]][[chain]][[sample]][["iV"]] = NULL
    obj.TF[["postList"]][[chain]][[sample]][["rho"]] = rho
    obj.TF[["postList"]][[chain]][[sample]][["sigma"]] = sigma^2
  }
}
obj.TF = alignPosterior(obj.TF)

#### Step 6. Visualize ####

#
# Plot sampled posteriors summaries from R only and TF
#
# obj.TF = obj.R #CHECK THIS

b.R = getPostEstimate(obj.R, "Beta")
b.TF = getPostEstimate(obj.TF, "Beta")
par(mfrow=c(ceiling(nc/2),min(2,nc-1)))
for(k in 1:obj.R$nc){
  plot(b.R$mean[k,], b.TF$mean[k,], main=paste(k, obj.R$covNames[k]))
  abline(0,1)
  # text(b.R$mean[k,], b.TF$mean[k,])
}
par(mfrow=c(1,1))
if(any(obj.R$distr[,2] != 0)){
  s.R = getPostEstimate(obj.R, "sigma")
  s.TF = getPostEstimate(obj.TF, "sigma")
  plot(s.R$mean, s.TF$mean, main="sigma")
  abline(0,1)
}
if(obj.R$nr > 0){
  omega.R = getPostEstimate(obj.R, "Omega")$mean
  omega.TF = getPostEstimate(obj.TF, "Omega")$mean
  # plot(crossprod(Lambda), omega.R, ylim=range(c(omega.R,omega.TF)), main="Omega")
  # points(crossprod(Lambda), omega.TF, col="blue")
  # plot(crossprod(Lambda)[-19,-19], omega.R[-19,-19], ylim=range(c(omega.R[-19,-19],omega.TF[-19,-19])), main="Omega")
  # points(crossprod(Lambda)[-19,-19], omega.TF[-19,-19], col="blue")
  plot(omega.R, omega.TF, main="Omega")
  abline(0,1)
  
  lambda.R = getPostEstimate(obj.R, "Lambda")$mean[1:2,]
  lambda.TF = getPostEstimate(obj.TF, "Lambda")$mean[1:2,]
  # plot(lambda.R[1,], lambda.TF[2,])
  # plot(lambda.R[2,], lambda.TF[1,])
}


obj.R.TF = obj.R
obj.R.TF$postList = c(obj.R$postList,obj.TF$postList)
obj.list = list(R=obj.R,TF=obj.TF,R.TF=obj.R.TF)

#beta, gamma, sigma
varVec = c("Beta","Gamma")
if(any(obj.R$distr[,2] != 0)) varVec = c(varVec, "Sigma")
for(variable in 1:length(varVec)){
  for(i in 1:3){
    mpost = convertToCodaObject(obj.list[[i]], Lambda=FALSE, Omega=FALSE, Psi=FALSE, Delta=FALSE, Eta=FALSE)
    mpost.var = mpost[[varVec[variable]]]
    psrf = gelman.diag(mpost.var, multivariate=FALSE)$psrf
    if(i == 1) {ma = psrf[,1]} else {ma = cbind(ma,psrf[,1])}
  }
  par(mfrow=c(2,1))
  print(paste(varVec[variable], "NaN", paste(apply(is.nan(ma), 2, sum),collapse=" "), sep=" - "))
  ma[is.nan(ma)] = 0.9
  vioplot(ma,names=names(obj.list),ylim=c(0.9,max(ma, na.rm=TRUE)),main=varVec[variable])
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=varVec[variable])
  par(mfrow=c(1,1))
}

#rho
if(!is.null(obj.R$C)){
  mat = rbind(unlist(getPostEstimate(obj.R, "rho")), unlist(getPostEstimate(obj.TF, "rho")))
  rownames(mat) = c("R", "TF")
  cat("rho posterior summary\n")
  print(mat)
}

#omega
maxOmega = 1000 #number of species pairs to be subsampled
nr = obj.list[[1]]$nr
for(k in seq_len(nr)){
  for(i in 1:3){
    mpost = convertToCodaObject(obj.list[[i]], Beta=FALSE, Lambda=FALSE, Psi=FALSE, Delta=FALSE, Eta=FALSE)
    tmp = mpost$Omega[[k]]
    z = dim(tmp[[1]])[2]
    if(z > maxOmega){
      if(i==1) sel = sample(1:z, size = maxOmega)
      for(j in 1:length(tmp)){
        tmp[[j]] = tmp[[j]][,sel]
      }
    }
    psrf = gelman.diag(tmp, multivariate = FALSE)$psrf
    if(i == 1) {ma = psrf[,1]} else {ma = cbind(ma,psrf[,1])}
  }
  par(mfrow=c(2,1))
  print(paste(varVec[variable], sprintf("Omega%d",k), paste(apply(is.nan(ma), 2, sum),collapse=" "), sep=" - "))
  print(paste(varVec[variable], sprintf("Omega%d",k), paste(apply(is.infinite(ma), 2, sum),collapse=" "), sep=" - "))
  ma[is.nan(ma) | is.infinite(ma)] = 0.9
  vioplot(ma,names=names(obj.list),ylim=c(0.9,max(ma)),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main=paste("omega",names(obj.list[[1]]$ranLevels)[k]))
  par(mfrow=c(1,1))
}

#reduced rank regression - effective beta
if(obj.R$ncRRR > 0){
  for(i in 1:3){
    BetaRRR_list = vector("list", nChains)
    for (chain in seq_len(nChains)) {
      BetaRRR_samples = vector("list", nSamples)
      for (sample in seq_len(nSamples)) {
        Beta = obj.list[[i]][["postList"]][[chain]][[sample]][["Beta"]]
        w = obj.list[[i]][["postList"]][[chain]][[sample]][["wRRR"]]
        BetaRRR_samples[[sample]] = as.vector(crossprod(w, Beta[-(1:obj.R$ncNRRR),]))
      }
      BetaRRR_list[[chain]] = mcmc(abind(BetaRRR_samples, along=0))
    }
    BetaRRR_mcmc.list = as.mcmc.list(BetaRRR_list)
    meanVal = colMeans(as.matrix(BetaRRR_mcmc.list))
    psrf = gelman.diag(BetaRRR_mcmc.list, multivariate=FALSE)$psrf
    if(i == 1) {ma = psrf[,1]; mv = meanVal} else {ma = cbind(ma,psrf[,1]); mv = cbind(mv,meanVal)}
  }
  plot(mv[,1], mv[,2])
  par(mfrow=c(2,1))
  vioplot(ma,names=names(obj.list),ylim=c(0.9,max(ma)),main="BetaRRR")
  vioplot(ma,names=names(obj.list),ylim=c(0.9,1.1),main="BetaRRR")
  par(mfrow=c(1,1))
}

mpost_R = convertToCodaObject(obj.R, Lambda=FALSE, Omega=FALSE, Psi=FALSE, Delta=FALSE, Eta=FALSE)$V
plot(mpost_R)
mpost_TF = convertToCodaObject(obj.TF, Lambda=FALSE, Omega=FALSE, Psi=FALSE, Delta=FALSE, Eta=FALSE)$V
plot(mpost_TF)
par(mfrow=c(1,1))

# psrf_R = gelman.diag(mpost_R, multivariate=FALSE)$psrf
# psrf_TF = gelman.diag(mpost_TF, multivariate=FALSE)$psrf
# plot(psrf_R[,1], psrf_TF[,1], type="n")
# text(psrf_R[,1], psrf_TF[,1], 1:nrow(psrf_R))



# library(truncnorm)
# for(cInd in 1:chain){
#   # ind = order(rowMeans(abs(obj.R$postList[[cInd]][[nSamples]]$Lambda[[1]])^2), decreasing=TRUE)[1:nc]
#   cInd2 = 1
#   ind = which(rowSums(obj.R$postList[[cInd2]][[nSamples]]$Psi[[1]]) > 0)
#   init_obj$initParList[[cInd]]$Eta[[1]] = obj.R$postList[[cInd2]][[nSamples]]$Eta[[1]][,ind,drop=FALSE]
#   init_obj$initParList[[cInd]]$Lambda[[1]] = obj.R$postList[[cInd2]][[nSamples]]$Lambda[[1]][ind,,drop=FALSE]
#   init_obj$initParList[[cInd]]$Delta[[1]] = 1+0*obj.R$postList[[cInd2]][[nSamples]]$Delta[[1]][ind,,drop=FALSE]
#   init_obj$initParList[[cInd]]$Psi[[1]] = 1+0*obj.R$postList[[cInd2]][[nSamples]]$Psi[[1]][ind,,drop=FALSE]
#   init_obj$initParList[[cInd]]$Alpha[[1]] = obj.R$postList[[cInd2]][[nSamples]]$Alpha[[1]][ind,drop=FALSE]
#   L = init_obj$initParList[[cInd]]$Eta[[1]] %*% init_obj$initParList[[cInd]]$Lambda[[1]]
#   lB = rep(-Inf, length(Y))
#   uB = rep(Inf, length(Y))
#   lB[Y] = 0
#   uB[!Y] = 0
#   z = rtruncnorm(length(Y), a=lB, b=uB, mean=L, sd=1)
#   init_obj$initParList[[cInd]]$Z = matrix(z, nrow(L), ncol(L))
# }
# write(to_json(init_obj), file = init_file_path)


