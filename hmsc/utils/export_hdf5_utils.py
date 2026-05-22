"""Posterior HDF5 export for Python-native workflows."""

from __future__ import annotations

import numpy as np


def save_chains_postList_to_hdf5(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to write HDF5 posterior outputs") from exc

    with h5py.File(postList_file_path, "w") as handle:
        handle.attrs["time"] = elapsedTime
        handle.attrs["nChains"] = nChains
        for param in ["Beta", "Gamma", "iV", "rhoInd", "sigma"]:
            handle.create_dataset(param, data=_stack_dense(postList, param))


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
