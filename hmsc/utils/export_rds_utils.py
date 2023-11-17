import ujson as json
import pandas as pd
import rdata
import pyreadr
import os


def read_r(fpath):

    def json_constructor(obj, attrs):
        assert len(obj) == 1
        return json.loads(str(obj[0]))

    data = rdata.parser.parse_file(fpath)
    data_dict = rdata.conversion.convert(
        data,
        {
            **rdata.conversion.DEFAULT_CLASS_MAP,
            "json": json_constructor,
        }
        )
    return data_dict


def load_model_from_rds(rds_file_path):
    init_obj = read_r(rds_file_path)
    return init_obj, init_obj["hM"]


def save_chains_postList_to_rds(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):

    json_data = {chain: {} for chain in range(nChains)}
    json_data["time"] = elapsedTime

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
            sample_data["Eta"] = [par.numpy().tolist() for par in params["Eta"]] if flag_save_eta else None
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