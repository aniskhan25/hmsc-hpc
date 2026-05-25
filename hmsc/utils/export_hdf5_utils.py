"""Posterior HDF5 export for Python-native workflows."""

from __future__ import annotations

import json

import numpy as np


def save_chains_postList_to_hdf5(
    postList,
    postList_file_path,
    nChains,
    elapsedTime=-1,
    flag_save_eta=True,
    metadata=None,
):
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write HDF5 posterior outputs") from exc

    with h5py.File(postList_file_path, "w") as handle:
        handle.attrs["time"] = elapsedTime
        handle.attrs["nChains"] = nChains
        if metadata is not None:
            handle.attrs["pyhmsc_metadata"] = json.dumps(metadata)
        for param in ["Beta", "Gamma", "iV", "rhoInd", "sigma"]:
            handle.create_dataset(param, data=_stack_dense(postList, param))
        random_group = handle.create_group("random_levels")
        n_levels = len(postList[0][0]["Eta"])
        for level in range(n_levels):
            level_group = random_group.create_group(str(level))
            for param in ["Eta", "Lambda", "Psi", "Delta", "AlphaInd"]:
                dataset_name = "Alpha" if param == "AlphaInd" else param
                level_group.create_dataset(dataset_name, data=_stack_random_level(postList, param, level))


def _stack_dense(postList, param):
    chains = []
    for chain in postList:
        draws = []
        for sample in chain:
            value = sample[param]
            if param == "rhoInd":
                value = value + 1
            draws.append(value.numpy())
        chains.append(np.stack(draws, axis=0))
    return np.stack(chains, axis=0)


def _stack_random_level(postList, param, level):
    chains = []
    for chain in postList:
        draws = []
        for sample in chain:
            value = sample[param][level]
            if param == "AlphaInd":
                value = value + 1
            draws.append(value.numpy())
        chains.append(np.stack(draws, axis=0))
    return np.stack(chains, axis=0)
