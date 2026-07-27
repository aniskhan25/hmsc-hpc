"""Training helpers for experimental Neural-HMSC prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from pyhmsc.neural.posterior_heads import (
    beta_negative_log_probability,
    gamma_negative_log_probability,
)
from pyhmsc.neural.simulator import (
    FixedEffectDataset,
    IidLatentEffectDataset,
    SpatialLatentEffectDataset,
    TraitEffectDataset,
)
from pyhmsc.serialization import read_compiled_model


@dataclass(frozen=True)
class FixedShapeTrainingData:
    """Tensor arrays for fixed-shape Beta posterior training."""

    X: np.ndarray
    Y: np.ndarray
    Beta: np.ndarray


@dataclass(frozen=True)
class VariableShapeTrainingData:
    """Padded arrays and masks for variable-shape Beta posterior work."""

    X: np.ndarray
    Y: np.ndarray
    Beta: np.ndarray
    site_mask: np.ndarray
    species_mask: np.ndarray


@dataclass(frozen=True)
class VariableDesignTrainingData:
    """Arrays and masks for variable site, species, and covariate dimensions."""

    X: np.ndarray
    Y: np.ndarray
    Beta: np.ndarray
    site_mask: np.ndarray
    species_mask: np.ndarray
    covariate_mask: np.ndarray
    covariate_names: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VariableDesignPredictiveAuxiliaryData:
    """Paired inference contexts and independent response-score tensors."""

    contexts: VariableDesignTrainingData
    heldouts: VariableDesignTrainingData
    context_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]


@dataclass(frozen=True)
class TraitEffectTrainingData:
    """Tensor arrays for fixed-shape trait-mediated Gamma posterior training."""

    X: np.ndarray
    Y: np.ndarray
    T: np.ndarray
    Beta: np.ndarray
    Gamma: np.ndarray


@dataclass(frozen=True)
class IidLatentTrainingData:
    """Tensor arrays for fixed-shape iid latent-factor posterior work."""

    X: np.ndarray
    Y: np.ndarray
    group_codes: np.ndarray
    Beta: np.ndarray
    Eta: np.ndarray
    Lambda: np.ndarray
    random_effect: np.ndarray


@dataclass(frozen=True)
class SpatialLatentTrainingData(IidLatentTrainingData):
    """Tensor arrays for full-spatial latent-factor posterior work."""

    coords: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(frozen=True)
class FixedShapeTrainingHistory:
    """Compact training history for fixed-shape prototype loops."""

    loss: list[float]
    beta_rmse: list[float]
    scale_mean: list[float]
    rank_mean_penalty: list[float] | None = None


def fixed_shape_training_data(
    datasets: Sequence[FixedEffectDataset],
) -> FixedShapeTrainingData:
    """Convert same-shape fixed-effect datasets to model-ready arrays."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    X_arrays = []
    Y_arrays = []
    beta_arrays = []
    expected_shape = None
    expected_covariates = None
    for dataset in datasets:
        covariates = [str(name) for name in dataset.truth_beta.index]
        if (
            not covariates
            or covariates[0] != "Intercept"
            or "Intercept" in covariates[1:]
        ):
            raise ValueError("truth_beta must contain one leading Intercept row")
        if expected_covariates is None:
            expected_covariates = covariates
        elif covariates != expected_covariates:
            raise ValueError(
                "all datasets must use the same fixed-effect covariate names"
            )
        predictors = covariates[1:]
        missing = [name for name in predictors if name not in dataset.X.columns]
        if missing:
            raise ValueError(f"dataset X is missing fixed-effect covariates: {missing}")
        columns = [np.ones(len(dataset.X), dtype=np.float32)]
        if predictors:
            columns.append(dataset.X[predictors].to_numpy(dtype=np.float32))
        design = np.column_stack(columns)
        Y = dataset.Y.to_numpy(dtype=np.float32)
        beta = dataset.truth_beta.loc[covariates].to_numpy(dtype=np.float32)
        shape = (design.shape, Y.shape, beta.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all datasets must have the same fixed shapes")
        X_arrays.append(design)
        Y_arrays.append(Y)
        beta_arrays.append(beta)
    return FixedShapeTrainingData(
        X=np.stack(X_arrays).astype(np.float32),
        Y=np.stack(Y_arrays).astype(np.float32),
        Beta=np.stack(beta_arrays).astype(np.float32),
    )


def variable_shape_training_data(
    datasets: Sequence[FixedEffectDataset],
) -> VariableShapeTrainingData:
    """Convert fixed-effect datasets to padded variable-shape arrays.

    Covariates must be identical across datasets. Sites and species are padded
    to the maximum shape in the batch.
    """
    if not datasets:
        raise ValueError("datasets must not be empty")
    covariates = [str(name) for name in datasets[0].truth_beta.index]
    if not covariates or covariates[0] != "Intercept":
        raise ValueError("truth_beta must contain one leading Intercept row")
    if "Intercept" in covariates[1:]:
        raise ValueError("truth_beta must contain one leading Intercept row")
    n_covariates = len(covariates)
    max_sites = max(len(dataset.X) for dataset in datasets)
    max_species = max(dataset.Y.shape[1] for dataset in datasets)
    batch = len(datasets)
    X = np.zeros((batch, max_sites, n_covariates), dtype=np.float32)
    Y = np.zeros((batch, max_sites, max_species), dtype=np.float32)
    Beta = np.zeros((batch, n_covariates, max_species), dtype=np.float32)
    site_mask = np.zeros((batch, max_sites), dtype=bool)
    species_mask = np.zeros((batch, max_species), dtype=bool)

    for idx, dataset in enumerate(datasets):
        dataset_covariates = [str(name) for name in dataset.truth_beta.index]
        if dataset_covariates != covariates:
            raise ValueError("all datasets must use the same ordered covariates")
        missing = [name for name in covariates[1:] if name not in dataset.X.columns]
        if missing:
            raise ValueError(f"dataset X is missing covariates: {missing}")
        n_sites = len(dataset.X)
        n_species = dataset.Y.shape[1]
        X[idx, :n_sites, :] = np.column_stack(
            [
                np.ones(n_sites, dtype=np.float32),
                dataset.X[covariates[1:]].to_numpy(dtype=np.float32),
            ]
        )
        Y[idx, :n_sites, :n_species] = dataset.Y.to_numpy(dtype=np.float32)
        Beta[idx, :, :n_species] = dataset.truth_beta.loc[covariates].to_numpy(
            dtype=np.float32
        )
        site_mask[idx, :n_sites] = True
        species_mask[idx, :n_species] = True

    return VariableShapeTrainingData(
        X=X,
        Y=Y,
        Beta=Beta,
        site_mask=site_mask,
        species_mask=species_mask,
    )


def variable_design_training_data(
    datasets: Sequence[FixedEffectDataset],
) -> VariableDesignTrainingData:
    """Pad fixed-effect datasets across sites, species, and design columns."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    names_by_dataset: list[tuple[str, ...]] = []
    for dataset in datasets:
        names = tuple(str(name) for name in dataset.truth_beta.index)
        if not names or names[0] != "Intercept" or "Intercept" in names[1:]:
            raise ValueError("truth_beta must contain one leading Intercept row")
        if len(set(names)) != len(names):
            raise ValueError("truth_beta covariate names must be unique")
        missing = [name for name in names[1:] if name not in dataset.X.columns]
        if missing:
            raise ValueError(f"dataset X is missing covariates: {missing}")
        names_by_dataset.append(names)

    max_sites = max(len(dataset.X) for dataset in datasets)
    max_species = max(dataset.Y.shape[1] for dataset in datasets)
    max_covariates = max(len(names) for names in names_by_dataset)
    batch = len(datasets)
    X = np.zeros((batch, max_sites, max_covariates), dtype=np.float32)
    Y = np.zeros((batch, max_sites, max_species), dtype=np.float32)
    Beta = np.zeros((batch, max_covariates, max_species), dtype=np.float32)
    site_mask = np.zeros((batch, max_sites), dtype=bool)
    species_mask = np.zeros((batch, max_species), dtype=bool)
    covariate_mask = np.zeros((batch, max_covariates), dtype=bool)

    for idx, (dataset, names) in enumerate(zip(datasets, names_by_dataset)):
        n_sites = len(dataset.X)
        n_species = dataset.Y.shape[1]
        n_covariates = len(names)
        columns = [np.ones(n_sites, dtype=np.float32)]
        if n_covariates > 1:
            columns.append(dataset.X[list(names[1:])].to_numpy(dtype=np.float32))
        X[idx, :n_sites, :n_covariates] = np.column_stack(columns)
        Y[idx, :n_sites, :n_species] = dataset.Y.to_numpy(dtype=np.float32)
        Beta[idx, :n_covariates, :n_species] = dataset.truth_beta.loc[
            list(names)
        ].to_numpy(dtype=np.float32)
        site_mask[idx, :n_sites] = True
        species_mask[idx, :n_species] = True
        covariate_mask[idx, :n_covariates] = True

    return VariableDesignTrainingData(
        X=X,
        Y=Y,
        Beta=Beta,
        site_mask=site_mask,
        species_mask=species_mask,
        covariate_mask=covariate_mask,
        covariate_names=tuple(names_by_dataset),
    )


def variable_design_predictive_auxiliary_data(
    contexts: Sequence[FixedEffectDataset],
    heldouts: Sequence[FixedEffectDataset],
) -> VariableDesignPredictiveAuxiliaryData:
    """Validate and pad independently generated predictive-score pairs."""
    if not contexts or not heldouts:
        raise ValueError("predictive auxiliary contexts and heldouts must not be empty")
    if len(contexts) != len(heldouts):
        raise ValueError("predictive auxiliary context/heldout counts differ")

    context_seeds = []
    heldout_seeds = []
    for index, (context, heldout) in enumerate(zip(contexts, heldouts)):
        context_names = tuple(str(name) for name in context.truth_beta.index)
        heldout_names = tuple(str(name) for name in heldout.truth_beta.index)
        if context_names != heldout_names:
            raise ValueError(
                f"predictive auxiliary covariate names differ at pair {index}"
            )
        if tuple(str(name) for name in context.Y.columns) != tuple(
            str(name) for name in heldout.Y.columns
        ):
            raise ValueError(
                f"predictive auxiliary species names differ at pair {index}"
            )
        if (
            len(context.X) != len(heldout.X)
            or context.Y.shape != heldout.Y.shape
            or context.truth_beta.shape != heldout.truth_beta.shape
        ):
            raise ValueError(f"predictive auxiliary shapes differ at pair {index}")
        if not np.array_equal(
            context.truth_beta.to_numpy(dtype=float),
            heldout.truth_beta.to_numpy(dtype=float),
        ):
            raise ValueError(
                f"predictive auxiliary coefficient truth differs at pair {index}"
            )
        context_seed = _required_dataset_seed(context, label="context", index=index)
        heldout_seed = _required_dataset_seed(heldout, label="heldout", index=index)
        if context_seed == heldout_seed:
            raise ValueError(
                f"predictive auxiliary pair {index} reused its context seed"
            )
        context_seeds.append(context_seed)
        heldout_seeds.append(heldout_seed)

    context_data = variable_design_training_data(contexts)
    heldout_data = variable_design_training_data(heldouts)
    for name in (
        "X",
        "Y",
        "Beta",
        "site_mask",
        "species_mask",
        "covariate_mask",
    ):
        if getattr(context_data, name).shape != getattr(heldout_data, name).shape:
            raise ValueError(f"predictive auxiliary padded {name} shapes differ")
    if context_data.covariate_names != heldout_data.covariate_names:
        raise ValueError("predictive auxiliary padded covariate names differ")
    if np.array_equal(context_data.Y, heldout_data.Y) and np.array_equal(
        context_data.X, heldout_data.X
    ):
        raise ValueError("predictive heldouts are not independent from contexts")
    return VariableDesignPredictiveAuxiliaryData(
        contexts=context_data,
        heldouts=heldout_data,
        context_seeds=tuple(context_seeds),
        heldout_seeds=tuple(heldout_seeds),
    )


def _required_dataset_seed(
    dataset: FixedEffectDataset, *, label: str, index: int
) -> int:
    if "seed" not in dataset.metadata:
        raise ValueError(
            f"predictive auxiliary {label} seed is missing at pair {index}"
        )
    seed = int(dataset.metadata["seed"])
    if seed < 0:
        raise ValueError(
            f"predictive auxiliary {label} seed is invalid at pair {index}"
        )
    return seed


def trait_effect_training_data(
    datasets: Sequence[TraitEffectDataset],
) -> TraitEffectTrainingData:
    """Convert same-shape trait-effect datasets to model-ready arrays."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    X_arrays = []
    Y_arrays = []
    T_arrays = []
    beta_arrays = []
    gamma_arrays = []
    expected_shape = None
    for dataset in datasets:
        covariate_names = [str(value) for value in dataset.truth_gamma.index]
        if not covariate_names or covariate_names[0] != "Intercept":
            raise ValueError("trait-effect Gamma covariates must start with Intercept")
        missing_covariates = [
            name for name in covariate_names[1:] if name not in dataset.X.columns
        ]
        if missing_covariates:
            raise ValueError(
                f"trait-effect dataset missing covariates: {missing_covariates}"
            )
        design = np.column_stack(
            [
                np.ones(len(dataset.X), dtype=np.float32),
                dataset.X[covariate_names[1:]].to_numpy(dtype=np.float32),
            ]
        )
        Y = dataset.Y.to_numpy(dtype=np.float32)
        trait_names = [str(value) for value in dataset.truth_gamma.columns]
        missing_traits = [
            name for name in trait_names if name not in dataset.trait_design.columns
        ]
        if missing_traits:
            raise ValueError(f"trait-effect dataset missing traits: {missing_traits}")
        T = dataset.trait_design[trait_names].to_numpy(dtype=np.float32)
        beta = dataset.truth_beta.loc[covariate_names].to_numpy(dtype=np.float32)
        gamma = dataset.truth_gamma.loc[covariate_names, trait_names].to_numpy(
            dtype=np.float32
        )
        shape = (design.shape, Y.shape, T.shape, beta.shape, gamma.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "all trait-effect datasets must have the same fixed shapes"
            )
        X_arrays.append(design)
        Y_arrays.append(Y)
        T_arrays.append(T)
        beta_arrays.append(beta)
        gamma_arrays.append(gamma)
    return TraitEffectTrainingData(
        X=np.stack(X_arrays).astype(np.float32),
        Y=np.stack(Y_arrays).astype(np.float32),
        T=np.stack(T_arrays).astype(np.float32),
        Beta=np.stack(beta_arrays).astype(np.float32),
        Gamma=np.stack(gamma_arrays).astype(np.float32),
    )


def iid_latent_training_data(
    datasets: Sequence[IidLatentEffectDataset],
) -> IidLatentTrainingData:
    """Convert same-shape iid latent datasets to model-ready arrays."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    X_arrays = []
    Y_arrays = []
    code_arrays = []
    beta_arrays = []
    eta_arrays = []
    lambda_arrays = []
    effect_arrays = []
    expected_shape = None
    for dataset in datasets:
        n_sites = len(dataset.X)
        design = np.column_stack(
            [
                np.ones(n_sites, dtype=np.float32),
                dataset.X[["x1", "x2"]].to_numpy(dtype=np.float32),
            ]
        )
        Y = dataset.Y.to_numpy(dtype=np.float32)
        codes = np.asarray(dataset.group_codes, dtype=np.int32)
        beta = dataset.truth_beta.loc[["Intercept", "x1", "x2"]].to_numpy(
            dtype=np.float32
        )
        eta = dataset.truth_eta.to_numpy(dtype=np.float32)
        loadings = dataset.truth_lambda.to_numpy(dtype=np.float32)
        random_effect = dataset.truth_random_effect.to_numpy(dtype=np.float32)
        shape = (
            design.shape,
            Y.shape,
            codes.shape,
            beta.shape,
            eta.shape,
            loadings.shape,
            random_effect.shape,
        )
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all iid latent datasets must have the same fixed shapes")
        if codes.min(initial=0) < 0 or codes.max(initial=0) >= eta.shape[0]:
            raise ValueError("group codes must index Eta rows")
        X_arrays.append(design)
        Y_arrays.append(Y)
        code_arrays.append(codes)
        beta_arrays.append(beta)
        eta_arrays.append(eta)
        lambda_arrays.append(loadings)
        effect_arrays.append(random_effect)
    return IidLatentTrainingData(
        X=np.stack(X_arrays).astype(np.float32),
        Y=np.stack(Y_arrays).astype(np.float32),
        group_codes=np.stack(code_arrays).astype(np.int32),
        Beta=np.stack(beta_arrays).astype(np.float32),
        Eta=np.stack(eta_arrays).astype(np.float32),
        Lambda=np.stack(lambda_arrays).astype(np.float32),
        random_effect=np.stack(effect_arrays).astype(np.float32),
    )


def spatial_latent_training_data(
    datasets: Sequence[SpatialLatentEffectDataset],
) -> SpatialLatentTrainingData:
    """Convert same-shape full-spatial latent datasets to model-ready arrays."""
    base = iid_latent_training_data(datasets)
    coords_arrays = []
    train_masks = []
    test_masks = []
    expected = None
    for dataset in datasets:
        coords = dataset.coords[["xcoord", "ycoord"]].to_numpy(dtype=np.float32)
        train_mask = np.asarray(dataset.train_mask, dtype=bool)
        test_mask = np.asarray(dataset.test_mask, dtype=bool)
        shape = (coords.shape, train_mask.shape, test_mask.shape)
        if expected is None:
            expected = shape
        elif shape != expected:
            raise ValueError(
                "all spatial latent datasets must have the same coordinate and split shapes"
            )
        if not train_mask.any() or not test_mask.any():
            raise ValueError(
                "spatial latent datasets require non-empty train and test masks"
            )
        coords_arrays.append(coords)
        train_masks.append(train_mask)
        test_masks.append(test_mask)
    return SpatialLatentTrainingData(
        X=base.X,
        Y=base.Y,
        group_codes=base.group_codes,
        Beta=base.Beta,
        Eta=base.Eta,
        Lambda=base.Lambda,
        random_effect=base.random_effect,
        coords=np.stack(coords_arrays).astype(np.float32),
        train_mask=np.stack(train_masks),
        test_mask=np.stack(test_masks),
    )


def compiled_trait_effect_training_data(
    init_json: str | Path,
    gamma_true: np.ndarray,
    *,
    beta_true: np.ndarray | None = None,
) -> TraitEffectTrainingData:
    """Build one trait-effect training item from compiled artifacts.

    This uses the compiler-emitted ``X``, ``Y``, and trait design ``T`` arrays.
    ``gamma_true`` is supplied by the simulation benchmark because compiled
    artifacts intentionally do not contain posterior truth.
    """
    metadata, arrays = read_compiled_model(init_json)
    required = ["X", "Y", "T"]
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"compiled artifacts missing arrays: {missing}")
    X = np.asarray(arrays["X"], dtype=np.float32)
    Y = np.asarray(arrays["Y"], dtype=np.float32)
    T = np.asarray(arrays["T"], dtype=np.float32)
    gamma = np.asarray(gamma_true, dtype=np.float32)
    expected_gamma = (X.shape[1], T.shape[1])
    if gamma.shape != expected_gamma:
        if gamma.shape == (expected_gamma[0], expected_gamma[1] + 1):
            gamma = gamma[:, 1:]
        else:
            raise ValueError(
                f"gamma_true shape {gamma.shape} does not match compiled Gamma shape {expected_gamma}"
            )
    if beta_true is None:
        beta = gamma @ T.T
    else:
        beta = np.asarray(beta_true, dtype=np.float32)
    expected_beta = (X.shape[1], Y.shape[1])
    if beta.shape != expected_beta:
        raise ValueError(
            f"beta_true shape {beta.shape} does not match compiled Beta shape {expected_beta}"
        )
    if metadata.get("dimensions", {}).get("n_traits") != T.shape[1]:
        raise ValueError("compiled metadata n_traits does not match T array")
    return TraitEffectTrainingData(
        X=X[None, ...],
        Y=Y[None, ...],
        T=T[None, ...],
        Beta=beta[None, ...],
        Gamma=gamma[None, ...],
    )


def train_fixed_shape_beta_model(
    model: tf.keras.Model,
    data: FixedShapeTrainingData,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    mse_weight: float = 0.25,
    verbose: int = 0,
) -> FixedShapeTrainingHistory:
    """Train a fixed-shape Beta posterior model against simulated truth."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dataset = (
        tf.data.Dataset.from_tensor_slices((data.X, data.Y, data.Beta))
        .shuffle(buffer_size=len(data.Beta), seed=123, reshuffle_each_iteration=True)
        .batch(batch_size)
    )
    history = {"loss": [], "beta_rmse": [], "scale_mean": []}

    def train_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        beta_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            posterior = model({"X": x_batch, "Y": y_batch}, training=True)
            nll = beta_negative_log_probability(posterior, beta_batch)
            mse = tf.reduce_mean(tf.square(beta_batch - posterior.mean))
            loss = nll + float(mse_weight) * mse
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(beta_batch - posterior.mean)))
        scale_mean = tf.reduce_mean(posterior.scale)
        return loss, rmse, scale_mean

    for epoch in range(int(epochs)):
        epoch_loss = []
        epoch_rmse = []
        epoch_scale = []
        for x_batch, y_batch, beta_batch in dataset:
            loss, rmse, scale_mean = train_step(x_batch, y_batch, beta_batch)
            epoch_loss.append(float(loss.numpy()))
            epoch_rmse.append(float(rmse.numpy()))
            epoch_scale.append(float(scale_mean.numpy()))
        history["loss"].append(float(np.mean(epoch_loss)))
        history["beta_rmse"].append(float(np.mean(epoch_rmse)))
        history["scale_mean"].append(float(np.mean(epoch_scale)))
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"loss={history['loss'][-1]:.4f} "
                f"beta_rmse={history['beta_rmse'][-1]:.4f} "
                f"scale_mean={history['scale_mean'][-1]:.4f}"
            )

    return FixedShapeTrainingHistory(
        loss=history["loss"],
        beta_rmse=history["beta_rmse"],
        scale_mean=history["scale_mean"],
    )


def train_trait_gamma_model(
    model: tf.keras.Model,
    data: TraitEffectTrainingData,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    mse_weight: float = 0.25,
    verbose: int = 0,
) -> FixedShapeTrainingHistory:
    """Train a trait-mediated Gamma posterior model against simulated truth."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rng = np.random.default_rng(123)
    history = {"loss": [], "beta_rmse": [], "scale_mean": []}

    def train_step(
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        t_batch: tf.Tensor,
        gamma_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            posterior = model({"X": x_batch, "Y": y_batch, "T": t_batch}, training=True)
            nll = gamma_negative_log_probability(posterior, gamma_batch)
            mse = tf.reduce_mean(tf.square(gamma_batch - posterior.mean))
            loss = nll + float(mse_weight) * mse
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        )
        rmse = tf.sqrt(tf.reduce_mean(tf.square(gamma_batch - posterior.mean)))
        scale_mean = tf.reduce_mean(posterior.scale)
        return loss, rmse, scale_mean

    for epoch in range(int(epochs)):
        epoch_loss = []
        epoch_rmse = []
        epoch_scale = []
        order = rng.permutation(len(data.Gamma))
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            x_batch = tf.convert_to_tensor(data.X[batch], dtype=tf.float32)
            y_batch = tf.convert_to_tensor(data.Y[batch], dtype=tf.float32)
            t_batch = tf.convert_to_tensor(data.T[batch], dtype=tf.float32)
            gamma_batch = tf.convert_to_tensor(data.Gamma[batch], dtype=tf.float32)
            loss, rmse, scale_mean = train_step(x_batch, y_batch, t_batch, gamma_batch)
            epoch_loss.append(float(loss.numpy()))
            epoch_rmse.append(float(rmse.numpy()))
            epoch_scale.append(float(scale_mean.numpy()))
        history["loss"].append(float(np.mean(epoch_loss)))
        history["beta_rmse"].append(float(np.mean(epoch_rmse)))
        history["scale_mean"].append(float(np.mean(epoch_scale)))
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"loss={history['loss'][-1]:.4f} "
                f"gamma_rmse={history['beta_rmse'][-1]:.4f} "
                f"scale_mean={history['scale_mean'][-1]:.4f}"
            )

    return FixedShapeTrainingHistory(
        loss=history["loss"],
        beta_rmse=history["beta_rmse"],
        scale_mean=history["scale_mean"],
    )
