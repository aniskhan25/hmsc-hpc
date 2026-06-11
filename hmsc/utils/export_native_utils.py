"""Load Python-native JSON+HDF5 initialization artifacts.

This loader is intentionally narrower than the RDS compatibility loader. It
supports Phase 2 fixed-effect models compiled by ``pyhmsc`` and returns the
internal objects expected by ``GibbsSampler``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

tfla = tf.linalg


_DISTR_CODES = {
    "normal": (1, 1),
    "gaussian": (1, 1),
    "probit": (2, 0),
    "bernoulli": (2, 0),
    "poisson": (3, 1),
}


def load_native_params(file_path, dtype=np.float64):
    metadata, arrays = load_compiled_json_hdf5(file_path, dtype)
    _validate_fixed_effect_schema(metadata, arrays)

    Y = arrays["Y"].astype(dtype)
    X = arrays["X"].astype(dtype)
    T = arrays.get("T", np.ones((Y.shape[1], 1), dtype=dtype)).astype(dtype)
    beta_init = arrays["Beta_init"].astype(dtype)
    n_chains = int(metadata["dimensions"]["n_chains"])
    n_sites, n_species = Y.shape
    n_covariates = X.shape[1]
    n_traits = T.shape[1]
    random_levels = metadata.get("random_levels", [])
    nr = len(random_levels)
    np_vec = np.array([level["n_levels"] for level in random_levels], dtype=int)

    model_dims = {
        "ny": n_sites,
        "ns": n_species,
        "nc": n_covariates,
        "nt": n_traits,
        "nr": nr,
        "np": np_vec,
        "ncsel": 0,
        "ncRRR": 0,
        "ncNRRR": 0,
        "ncORRR": 0,
        "nuRRR": 0,
    }

    C = arrays.get("C")
    if C is None:
        eC, VC = None, None
    else:
        C = C.astype(dtype)
        eC, VC = np.linalg.eigh(C)

    model_data = {
        "Y": Y,
        "Yo": np.logical_not(np.isnan(Y)),
        "X": X,
        "T": T,
        "C": C,
        "eC": eC,
        "VC": VC,
        "rhoGroup": np.zeros(n_covariates, dtype=int),
        "Pi": arrays.get("Pi", np.zeros((n_sites, 0), dtype=int)).astype(int),
        "distr": _distribution_matrix(metadata["distribution"], n_species),
        "XSel": [],
        "XRRR": np.zeros((n_sites, 0), dtype=dtype),
    }

    beta_prior = metadata.get("priors", {}).get("Beta", {})
    beta_variance = dtype(beta_prior.get("variance", 100.0))
    gamma_size = n_traits * n_covariates
    prior_hyperparams = {
        "mGamma": np.zeros(gamma_size, dtype=dtype),
        "UGamma": np.eye(gamma_size, dtype=dtype) * beta_variance,
        "iUGamma": np.eye(gamma_size, dtype=dtype) / beta_variance,
        "f0": dtype(n_covariates + 2),
        "V0": np.eye(n_covariates, dtype=dtype),
        "rhopw": np.array([[0.0, 1.0], [0.5, 1.0], [0.9, 1.0]], dtype=dtype),
        "aSigma": np.ones(n_species, dtype=dtype),
        "bSigma": np.ones(n_species, dtype=dtype),
        "nuRRR": dtype(0),
        "a1RRR": dtype(0),
        "b1RRR": dtype(0),
        "a2RRR": dtype(0),
        "b2RRR": dtype(0),
    }

    rL_hyperparams = [_random_level_hyperparams(level, arrays, dtype) for level in random_levels]
    init_par_list = [
        _fixed_effect_init_params(
            beta_init[chain],
            n_covariates,
            n_species,
            n_traits,
            dtype,
            random_levels=random_levels,
            arrays=arrays,
            chain=chain,
        )
        for chain in range(n_chains)
    ]
    for params in init_par_list:
        params["Xeff"] = tf.constant(X, dtype=dtype)
    return model_dims, model_data, prior_hyperparams, None, rL_hyperparams, init_par_list, n_chains


def load_compiled_json_hdf5(file_path, dtype=np.float64):
    path = Path(file_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install h5py to read Python-native compiled model inputs") from exc

    arrays = {}
    for name, ref in metadata.get("arrays", {}).items():
        file_part, dataset = _split_hdf5_ref(ref)
        with h5py.File(path.parent / file_part, "r") as handle:
            arrays[name] = np.asarray(handle[dataset], dtype=dtype)
    return metadata, arrays


def _fixed_effect_init_params(
    beta,
    n_covariates,
    n_species,
    n_traits,
    dtype,
    random_levels=None,
    arrays=None,
    chain=0,
):
    beta_tensor = tf.constant(beta, dtype=dtype)
    random_levels = random_levels or []
    arrays = arrays or {}
    eta_list = []
    lambda_list = []
    psi_list = []
    delta_list = []
    alpha_list = []
    for level in random_levels:
        prefix = level["array_prefix"]
        eta_list.append(tf.constant(arrays[f"{prefix}_Eta_init"][chain], dtype=dtype))
        lambda_list.append(tf.constant(arrays[f"{prefix}_Lambda_init"][chain], dtype=dtype))
        psi_list.append(tf.constant(arrays[f"{prefix}_Psi_init"][chain], dtype=dtype))
        delta_list.append(tf.constant(arrays[f"{prefix}_Delta_init"][chain], dtype=dtype))
        alpha_list.append(tf.cast(tf.constant(arrays[f"{prefix}_Alpha_init"][chain]), tf.int32) - 1)
    return {
        "Z": None,
        "Beta": beta_tensor,
        "Gamma": tf.zeros((n_covariates, n_traits), dtype=dtype),
        "iV": tf.eye(n_covariates, dtype=dtype),
        "rhoInd": tf.zeros((n_covariates,), dtype=tf.int32),
        "sigma": tf.ones((n_species,), dtype=dtype),
        "Lambda": lambda_list,
        "Psi": psi_list,
        "Delta": delta_list,
        "Eta": eta_list,
        "AlphaInd": alpha_list,
        "BetaSel": [],
        "PsiRRR": tf.zeros((0, 0), dtype=dtype),
        "DeltaRRR": tf.zeros((0,), dtype=dtype),
        "wRRR": tf.zeros((0, 0), dtype=dtype),
        "Xeff": tf.constant([], dtype=dtype),  # replaced by caller
    }


def _validate_fixed_effect_schema(metadata, arrays):
    if metadata.get("format") != "pyhmsc-json-hdf5":
        raise ValueError("Unsupported native input format")
    capabilities = metadata.get("capabilities", {})
    required_arrays = {"Y", "X", "Beta_init"}
    missing = required_arrays.difference(arrays)
    if missing:
        raise ValueError(f"Native input missing arrays: {sorted(missing)}")
    Y, X, beta_init = arrays["Y"], arrays["X"], arrays["Beta_init"]
    dims = metadata.get("dimensions", {})
    expected = (
        int(dims["n_sites"]),
        int(dims["n_species"]),
        int(dims["n_covariates"]),
        int(dims["n_chains"]),
    )
    if Y.shape != expected[:2]:
        raise ValueError(f"Y shape {Y.shape} does not match metadata {expected[:2]}")
    if X.shape != (expected[0], expected[2]):
        raise ValueError(f"X shape {X.shape} does not match metadata {(expected[0], expected[2])}")
    if "T" in arrays and arrays["T"].shape != (expected[1], int(metadata["dimensions"].get("n_traits", 1))):
        raise ValueError("T shape does not match metadata")
    if "C" in arrays and arrays["C"].shape != (expected[1], expected[1]):
        raise ValueError("C shape does not match species dimensions")
    if beta_init.shape != (expected[3], expected[2], expected[1]):
        raise ValueError(
            "Beta_init shape "
            f"{beta_init.shape} does not match metadata {(expected[3], expected[2], expected[1])}"
        )
    random_levels = metadata.get("random_levels", [])
    if random_levels:
        if "Pi" not in arrays:
            raise ValueError("Native random-level input missing Pi array")
        if arrays["Pi"].shape != (expected[0], len(random_levels)):
            raise ValueError("Pi shape does not match random level metadata")
        for level in random_levels:
            prefix = level["array_prefix"]
            required = [
                f"{prefix}_Eta_init",
                f"{prefix}_Lambda_init",
                f"{prefix}_Psi_init",
                f"{prefix}_Delta_init",
                f"{prefix}_Alpha_init",
            ]
            missing = [name for name in required if name not in arrays]
            if missing:
                raise ValueError(f"Native random-level input missing arrays: {missing}")
            if level.get("type") == "spatial_full" and f"{prefix}_distMat" not in arrays:
                raise ValueError("Native full spatial random level missing distMat array")
            if level.get("type") in {"spatial_gpp", "gpp"}:
                missing_gpp = [
                    name
                    for name in [f"{prefix}_distMat12", f"{prefix}_distMat22"]
                    if name not in arrays
                ]
                if missing_gpp:
                    raise ValueError(f"Native GPP spatial random level missing arrays: {missing_gpp}")
            if int(level.get("xDim", 0)) > 0 and f"{prefix}_xMat" not in arrays:
                raise ValueError("Native random-slope input missing xMat array")


def _distribution_matrix(distribution, n_species):
    key = str(distribution).lower()
    if key not in _DISTR_CODES:
        raise ValueError(
            f"Unsupported native distribution {distribution!r}; expected one of "
            f"{sorted(_DISTR_CODES)}"
        )
    return np.tile(np.array(_DISTR_CODES[key], dtype=int), (n_species, 1))


def _split_hdf5_ref(ref):
    if ":/" not in ref:
        raise ValueError(f"Invalid HDF5 array reference {ref!r}")
    file_part, dataset = ref.split(":", 1)
    return file_part, dataset


def _random_level_hyperparams(level, arrays, dtype):
    params = {
        "nu": dtype(level.get("nu", 3.0)),
        "a1": dtype(level.get("a1", 2.0)),
        "b1": dtype(level.get("b1", 1.0)),
        "a2": dtype(level.get("a2", 3.0)),
        "b2": dtype(level.get("b2", 1.0)),
        "nfMin": int(level.get("nfMin", level.get("nf", 1))),
        "nfMax": int(level.get("nfMax", max(level.get("nf", 1), 4))),
        "sDim": 0,
        "xDim": int(level.get("xDim", 0)),
    }
    if params["xDim"] > 0:
        params["xMat"] = np.asarray(arrays[f"{level['array_prefix']}_xMat"], dtype=dtype)
    if level.get("type") == "spatial_full":
        alphapw = np.asarray(level.get("alphapw", [[1.0, 1.0]]), dtype=dtype)
        dist = np.asarray(arrays[f"{level['array_prefix']}_distMat"], dtype=dtype)
        Wg = _spatial_full_W(dist, alphapw, dtype)
        iWg = []
        detWg = []
        for W in Wg:
            W = W + np.eye(W.shape[0], dtype=dtype) * dtype(1e-8)
            LW = np.linalg.cholesky(W)
            invW = np.linalg.solve(LW.T, np.linalg.solve(LW, np.eye(W.shape[0], dtype=dtype)))
            iWg.append(invW)
            detWg.append(2 * np.sum(np.log(np.diag(LW))))
        params.update(
            {
                "sDim": 2,
                "spatialMethod": "Full",
                "alphapw": alphapw,
                "iWg": np.stack(iWg, axis=0).astype(dtype),
                "detWg": np.asarray(detWg, dtype=dtype),
            }
        )
    elif level.get("type") in {"spatial_gpp", "gpp"}:
        alphapw = np.asarray(level.get("alphapw", [[1.0, 1.0]]), dtype=dtype)
        dist12 = np.asarray(arrays[f"{level['array_prefix']}_distMat12"], dtype=dtype)
        dist22 = np.asarray(arrays[f"{level['array_prefix']}_distMat22"], dtype=dtype)
        params.update(_spatial_gpp_params(dist12, dist22, alphapw, dtype))
    return params


def _spatial_full_W(dist, alphapw, dtype):
    if dist.size == 0:
        return np.zeros((alphapw.shape[0], 0, 0), dtype=dtype)
    W = []
    for alpha in alphapw[:, 0]:
        scale = alpha if alpha > 0 else 1.0
        W.append(np.exp(-dist / scale))
    return np.stack(W, axis=0).astype(dtype)


def _spatial_gpp_params(dist12, dist22, alphapw, dtype):
    g_count = alphapw.shape[0]
    n_knots = dist22.shape[0]
    w12 = []
    w22 = []
    for alpha in alphapw[:, 0]:
        scale = alpha if alpha > 0 else 1.0
        w12.append(np.exp(-dist12 / scale))
        w22.append(np.exp(-dist22 / scale))
    W12 = np.stack(w12, axis=0).astype(dtype)
    W22 = np.stack(w22, axis=0).astype(dtype)
    idD = []
    idDW12 = []
    F = []
    iF = []
    detD = []
    for g in range(g_count):
        W22_g = W22[g] + np.eye(n_knots, dtype=dtype) * dtype(1e-8)
        LW22 = np.linalg.cholesky(W22_g)
        iW22 = np.linalg.solve(LW22.T, np.linalg.solve(LW22, np.eye(n_knots, dtype=dtype)))
        dD = 1 - np.einsum("ik,kh,ih->i", W12[g], iW22, W12[g])
        dD = np.maximum(dD, np.finfo(dtype).eps)
        idD_g = dD ** -1
        F_g = W22_g + np.einsum("ik,i,ih->kh", W12[g], idD_g, W12[g])
        LF = np.linalg.cholesky(F_g + np.eye(n_knots, dtype=dtype) * dtype(1e-8))
        idD.append(idD_g)
        idDW12.append(idD_g[:, None] * W12[g])
        F.append(F_g)
        iF.append(np.linalg.solve(LF.T, np.linalg.solve(LF, np.eye(n_knots, dtype=dtype))))
        detD.append(np.sum(np.log(dD)) - 2 * np.sum(np.log(np.diag(LW22))) + 2 * np.sum(np.log(np.diag(LF))))
    return {
        "sDim": 2,
        "spatialMethod": "GPP",
        "alphapw": alphapw,
        "nK": int(n_knots),
        "idDg": np.stack(idD, axis=0).astype(dtype),
        "idDW12g": np.stack(idDW12, axis=0).astype(dtype),
        "Fg": np.stack(F, axis=0).astype(dtype),
        "iFg": np.stack(iF, axis=0).astype(dtype),
        "detDg": np.asarray(detD, dtype=dtype),
    }
