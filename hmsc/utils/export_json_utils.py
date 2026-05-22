import ujson as json
import pandas as pd
import os


def load_model_from_json(json_file_path):

    with open(json_file_path) as json_file:
        hmsc_obj = json.load(json_file)

    return hmsc_obj, hmsc_obj.get("hM")


def load_model_from_rds(rds_file_path):
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr to read RDS compatibility inputs") from exc

    long_str = pyreadr.read_r(rds_file_path)
    hmsc_obj = json.loads(long_str[None][None][0])
    
    return hmsc_obj, hmsc_obj.get("hM")


def save_postList_to_json(postList, postList_file_path, chain):

    print("Start dumping.")

    json_data = {}

    for i in range(len(postList)):
        sample_data = {}
        params = postList[i]
        
        sample_data["Beta"] = params["Beta"].numpy().tolist()
        sample_data["BetaSel"] = [par.numpy().tolist() for par in params["BetaSel"]]
        sample_data["Gamma"] = params["Gamma"].numpy().tolist()
        sample_data["iV"] = params["iV"].numpy().tolist()
        sample_data["rhoInd"] = (params["rhoInd"]+1).numpy().tolist()
        sample_data["sigma"] = params["sigma"].numpy().tolist()
        
        sample_data["Lambda"] = [par.numpy().tolist() for par in params["Lambda"]]
        sample_data["Psi"] = [par.numpy().tolist() for par in params["Psi"]]
        sample_data["Delta"] = [par.numpy().tolist() for par in params["Delta"]]
        sample_data["Eta"] = [par.numpy().tolist() for par in params["Eta"]]
        sample_data["Alpha"] = [(par+1).numpy().tolist() for par in params["AlphaInd"]]
        
        if params["wRRR"] is not None:
          sample_data["wRRR"] = params["wRRR"].numpy().tolist()
          sample_data["PsiRRR"] = params["PsiRRR"].numpy().tolist()
          sample_data["DeltaRRR"] = params["DeltaRRR"].numpy().tolist()
        else:
          sample_data["wRRR"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

        json_data[i] = sample_data

    postList_file_path = (
        os.path.splitext(postList_file_path)[0] + "_" + str(chain+1) + ".json"
    )
    print("Dumping, chain %d" % chain)
    with open(postList_file_path, "w") as fp:
        json.dump(json_data, fp)


def save_chains_postList_to_json(postList, postList_file_path, nChains):

    json_data = {chain: {} for chain in range(nChains)}

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}
            params = postList[chain][i]

            sample_data["Beta"] = params["Beta"].numpy().tolist()
            sample_data["BetaSel"] = [par.numpy().tolist() for par in params["BetaSel"]]
            sample_data["Gamma"] = params["Gamma"].numpy().tolist()
            sample_data["iV"] = params["iV"].numpy().tolist()
            sample_data["rhoInd"] = (params["rhoInd"]+1).numpy().tolist()
            sample_data["sigma"] = params["sigma"].numpy().tolist()
            
            sample_data["Lambda"] = [par.numpy().tolist() for par in params["Lambda"]]
            sample_data["Psi"] = [par.numpy().tolist() for par in params["Psi"]]
            sample_data["Delta"] = [par.numpy().tolist() for par in params["Delta"]]
            sample_data["Eta"] = [par.numpy().tolist() for par in params["Eta"]] 
            sample_data["Alpha"] = [(par+1).numpy().tolist() for par in params["AlphaInd"]]
            
            if params["wRRR"] is not None:
              sample_data["wRRR"] = params["wRRR"].numpy().tolist()
              sample_data["PsiRRR"] = params["PsiRRR"].numpy().tolist()
              sample_data["DeltaRRR"] = params["DeltaRRR"].numpy().tolist()
            else:
              sample_data["wRRR"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    with open(postList_file_path, "w") as fp:
        json.dump(json_data, fp)


def save_chains_postList_to_rds(postList, postList_file_path, nChains):
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr to write RDS compatibility outputs") from exc

    json_data = {chain: {} for chain in range(nChains)}

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}
            params = postList[chain][i]

            sample_data["Beta"] = params["Beta"].numpy().tolist()
            sample_data["BetaSel"] = [par.numpy().tolist() for par in params["BetaSel"]]
            sample_data["Gamma"] = params["Gamma"].numpy().tolist()
            sample_data["iV"] = params["iV"].numpy().tolist()
            sample_data["rhoInd"] = (params["rhoInd"]+1).numpy().tolist()
            sample_data["sigma"] = params["sigma"].numpy().tolist()
            
            sample_data["Lambda"] = [par.numpy().tolist() for par in params["Lambda"]]
            sample_data["Psi"] = [par.numpy().tolist() for par in params["Psi"]]
            sample_data["Delta"] = [par.numpy().tolist() for par in params["Delta"]]
            sample_data["Eta"] = [par.numpy().tolist() for par in params["Eta"]] 
            sample_data["Alpha"] = [(par+1).numpy().tolist() for par in params["AlphaInd"]]
            
            if params["wRRR"] is not None:
              sample_data["wRRR"] = params["wRRR"].numpy().tolist()
              sample_data["PsiRRR"] = params["PsiRRR"].numpy().tolist()
              sample_data["DeltaRRR"] = params["DeltaRRR"].numpy().tolist()
            else:
              sample_data["wRRR"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    json_str = json.dumps(json_data)
    
    pyreadr.write_rds(postList_file_path, pd.DataFrame([[json_str]]), compress="gzip")
