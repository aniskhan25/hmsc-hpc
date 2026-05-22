import numpy as np
import ujson as json
import pandas as pd


def load_model_from_rds(rds_file_path):
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr to read RDS compatibility inputs") from exc

    long_str = pyreadr.read_r(rds_file_path)
    hmsc_obj = json.loads(long_str[None][None][0])
    
    return hmsc_obj, hmsc_obj.get("hM")


def save_chains_postList_to_rds(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("Install pyreadr to write RDS compatibility outputs") from exc

    json_data = {chain: {} for chain in range(nChains)}
    json_data["time"] = elapsedTime

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}
            params = postList[chain][i]

            sample_data["Beta"] = params["Beta"].numpy().tolist()
            sample_data["BetaSel"] = dict(zip(np.arange(len(params["BetaSel"])), [par.numpy().tolist() for par in params["BetaSel"]]))
            sample_data["Gamma"] = params["Gamma"].numpy().tolist()
            sample_data["iV"] = params["iV"].numpy().tolist()
            sample_data["rhoInd"] = (params["rhoInd"]+1).numpy().tolist()
            sample_data["sigma"] = params["sigma"].numpy().tolist()
            
            sample_data["Lambda"] = dict(zip(np.arange(len(params["AlphaInd"])), [par.numpy().tolist() for par in params["Lambda"]]))
            sample_data["Psi"] = dict(zip(np.arange(len(params["AlphaInd"])), [par.numpy().tolist() for par in params["Psi"]]))
            sample_data["Delta"] = dict(zip(np.arange(len(params["AlphaInd"])), [par.numpy().tolist() for par in params["Delta"]]))
            sample_data["Eta"] = dict(zip(np.arange(len(params["AlphaInd"])), [par.numpy().tolist() for par in params["Eta"]])) if flag_save_eta else None
            sample_data["Alpha"] = dict(zip(np.arange(len(params["AlphaInd"])), [(par+1).numpy().tolist() for par in params["AlphaInd"]]))
            
            if params["wRRR"] is not None:
              sample_data["wRRR"] = params["wRRR"].numpy().tolist()
              sample_data["PsiRRR"] = params["PsiRRR"].numpy().tolist()
              sample_data["DeltaRRR"] = params["DeltaRRR"].numpy().tolist()
            else:
              sample_data["wRRR"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    json_str = json.dumps(json_data)
    
    pyreadr.write_rds(postList_file_path, pd.DataFrame([[json_str]]), compress="gzip")
