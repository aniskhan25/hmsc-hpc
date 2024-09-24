import warnings

import rdata
import tensorflow as tf
import xarray as xr


def convert_to_numpy(obj):
    if isinstance(obj, xr.DataArray):
        return obj.to_numpy()

    if isinstance(obj, tf.Tensor):
        return obj.numpy()

    if isinstance(obj, dict):
        new = {}
        for key, value in obj.items():
            new[key] = convert_to_numpy(value)
        return new

    if isinstance(obj, list):
        new = []
        for value in obj:
            new.append(convert_to_numpy(value))
        return new

    return obj


def load_model_from_rds(rds_file_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
            message=".*Missing constructor for R class.*",
            category=UserWarning)
        init_obj = rdata.read_rds(rds_file_path)

    init_obj = convert_to_numpy(init_obj)
    return init_obj, init_obj["hM"]


def save_chains_postList_to_rds(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):
    data = {}
    output_list = convert_to_numpy(postList)

    for i in range(len(output_list)):
        for j in range(len(output_list[i])):
            item = output_list[i][j]

            # Convert from zero- to one-based indices
            item["rhoInd"] += 1
            for k in range(len(item["AlphaInd"])):
                item["AlphaInd"][k] += 1

            # Compatibility with HMSC-R / duplicate AlphaInd to Alpha
            item["Alpha"] = item["AlphaInd"]

            # Remove eta if requested
            if not flag_save_eta:
                item["Eta"] = None

    data["list"] = output_list
    data["time"] = elapsedTime
    rdata.write_rds(postList_file_path, data)
