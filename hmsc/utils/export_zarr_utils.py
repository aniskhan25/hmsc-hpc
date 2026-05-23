"""Optional Zarr posterior export."""

from __future__ import annotations

from hmsc.utils.export_hdf5_utils import _stack_dense, _stack_random_level


def save_chains_postList_to_zarr(postList, postList_file_path, nChains, elapsedTime=-1, flag_save_eta=True):
    try:
        import zarr  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install zarr to write Zarr posterior outputs") from exc

    root = zarr.open_group(str(postList_file_path), mode="w")
    root.attrs["time"] = float(elapsedTime)
    root.attrs["nChains"] = int(nChains)
    for param in ["Beta", "Gamma", "iV", "rhoInd", "sigma"]:
        root.create_array(param, data=_stack_dense(postList, param), overwrite=True)
    random_group = root.create_group("random_levels", overwrite=True)
    n_levels = len(postList[0][0]["Eta"])
    for level in range(n_levels):
        level_group = random_group.create_group(str(level), overwrite=True)
        for param in ["Eta", "Lambda", "Psi", "Delta", "AlphaInd"]:
            dataset_name = "Alpha" if param == "AlphaInd" else param
            level_group.create_array(
                dataset_name,
                data=_stack_random_level(postList, param, level),
                overwrite=True,
            )
