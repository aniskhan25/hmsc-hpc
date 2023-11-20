import ujson as json
import pandas as pd
import rdata
import pyreadr
import os


def read_r(fpath):

    def version_constructor(obj, attrs):
        assert len(obj) == 1
        return tuple(obj[0])

    def json_constructor(obj, attrs):
        assert len(obj) == 1
        return json.loads(str(obj[0]))


    # Fix a bug in default intseq_constructor
    def compact_intseq_constructor(state):
        import numpy as np

        info = rdata.parser.RObjectInfo(
            type=rdata.parser.RObjectType.INT,
            object=False,
            attributes=False,
            tag=False,
            gp=0,
            reference=0,
        )
        n = int(state.value[0])
        start = int(state.value[1])
        step = int(state.value[2])
        stop = start + (n - 1) * step
        value = np.array(range(start, stop + 1, step))
        return info, value

    altrep_constructor_dict = {**rdata.parser.DEFAULT_ALTREP_MAP}
    altrep_constructor_dict[b"compact_intseq"] = compact_intseq_constructor
    altrep_constructor_dict[b"compact_realseq"] = lambda state: 1/0  # realseq is also buggy

    data = rdata.parser.parse_file(
        fpath,
        altrep_constructor_dict=altrep_constructor_dict,
    )
    # End of bug fix
    # data = rdata.parser.parse_file(fpath)

    data_dict = rdata.conversion.convert(
        data,
        {
            **rdata.conversion.DEFAULT_CLASS_MAP,
            "package_version": version_constructor,
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