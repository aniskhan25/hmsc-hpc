"""Experimental Neural-HMSC model prototypes."""

from __future__ import annotations

import tensorflow as tf

from pyhmsc.neural.posterior_heads import (
    BetaPosterior,
    DiagonalNormalBetaHead,
    DiagonalNormalGammaHead,
    FullCovarianceNormalBetaHead,
    GammaPosterior,
    IidLatentPosterior,
)


class FixedShapeBetaPosteriorModel(tf.keras.Model):
    """Fixed-shape amortized posterior model for Gaussian fixed-effect Beta."""

    def __init__(
        self,
        n_sites: int,
        n_covariates: int,
        n_species: int,
        hidden_units: tuple[int, ...] = (64, 64),
        min_scale: float = 1e-3,
        posterior_family: str = "diagonal_normal",
        distribution: str = "normal",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        if distribution not in {"normal", "probit", "poisson"}:
            raise ValueError("distribution must be 'normal', 'probit', or 'poisson'")
        self.distribution = str(distribution)
        if posterior_family not in {"diagonal_normal", "full_covariance_normal"}:
            raise ValueError("posterior_family must be 'diagonal_normal' or 'full_covariance_normal'")
        self.posterior_family = str(posterior_family)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]
        head_type = (
            FullCovarianceNormalBetaHead
            if self.posterior_family == "full_covariance_normal"
            else DiagonalNormalBetaHead
        )
        self.head = head_type(
            n_covariates=self.n_covariates,
            n_species=self.n_species,
            min_scale=min_scale,
        )

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | dict[str, tf.Tensor]) -> BetaPosterior:
        if isinstance(inputs, dict):
            design = inputs["X"]
            response = inputs["Y"]
        else:
            design, response = inputs
        design = tf.cast(design, tf.float32)
        response = tf.cast(response, tf.float32)
        _assert_fixed_shape(design, response, self.n_sites, self.n_covariates, self.n_species)

        feature_response = tf.math.log1p(response) if self.distribution == "poisson" else response
        xty = tf.einsum("bnk,bns->bks", design, feature_response) / tf.cast(self.n_sites, tf.float32)
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(self.n_sites, tf.float32)
        ridge = _ridge_beta_estimate(xtx, xty)
        y_mean = tf.reduce_mean(feature_response, axis=1)
        y_sd = tf.math.reduce_std(feature_response, axis=1)
        features = tf.concat(
            [
                tf.reshape(xty, (tf.shape(design)[0], -1)),
                tf.reshape(xtx, (tf.shape(design)[0], -1)),
                tf.reshape(ridge, (tf.shape(design)[0], -1)),
                y_mean,
                y_sd,
            ],
            axis=-1,
        )
        for layer in self.encoder_layers:
            features = layer(features)
        residual = self.head(features)
        return BetaPosterior(
            mean=ridge + residual.mean,
            scale=residual.scale,
            scale_tril=residual.scale_tril,
        )


class VariableShapeBetaPosteriorModel(tf.keras.Model):
    """Masked variable-site/species posterior model for Gaussian fixed effects."""

    def __init__(
        self,
        n_covariates: int = 3,
        hidden_units: tuple[int, ...] = (48, 48),
        min_scale: float = 1e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_covariates = int(n_covariates)
        self.min_scale = float(min_scale)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]
        self.projection = tf.keras.layers.Dense(
            2 * self.n_covariates,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, inputs: dict[str, tf.Tensor]) -> BetaPosterior:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        site_mask = tf.cast(inputs["site_mask"], tf.float32)
        species_mask = tf.cast(inputs["species_mask"], tf.float32)
        _assert_variable_shape(design, response, site_mask, species_mask, self.n_covariates)

        site_weights = site_mask[:, :, None]
        masked_design = design * site_weights
        masked_response = response * site_weights * species_mask[:, None, :]
        site_count = tf.maximum(tf.reduce_sum(site_mask, axis=1), 1.0)
        xty = tf.einsum("bnk,bns->bks", masked_design, masked_response) / site_count[:, None, None]
        xtx = tf.einsum("bnk,bnl->bkl", masked_design, masked_design) / site_count[:, None, None]
        ridge = _ridge_beta_estimate(xtx, xty)
        y_mean = tf.reduce_sum(masked_response, axis=1) / site_count[:, None]
        centered = (response - y_mean[:, None, :]) * site_weights * species_mask[:, None, :]
        y_sd = tf.sqrt(tf.reduce_sum(tf.square(centered), axis=1) / site_count[:, None] + 1e-8)
        xtx_features = tf.tile(
            tf.reshape(xtx, (tf.shape(design)[0], 1, self.n_covariates * self.n_covariates)),
            [1, tf.shape(response)[2], 1],
        )
        species_features = tf.concat(
            [
                tf.transpose(xty, [0, 2, 1]),
                tf.transpose(ridge, [0, 2, 1]),
                y_mean[:, :, None],
                y_sd[:, :, None],
                species_mask[:, :, None],
                xtx_features,
            ],
            axis=-1,
        )
        features = species_features
        for layer in self.encoder_layers:
            features = layer(features)
        raw = self.projection(features)
        mean_raw, scale_raw = tf.split(raw, 2, axis=-1)
        residual_mean = tf.transpose(mean_raw, [0, 2, 1])
        residual_scale = tf.transpose(tf.nn.softplus(scale_raw) + self.min_scale, [0, 2, 1])
        mask = species_mask[:, None, :]
        return BetaPosterior(
            mean=(ridge + residual_mean) * mask,
            scale=residual_scale * mask,
        )


class TraitGammaPosteriorModel(tf.keras.Model):
    """Fixed-shape trait-mediated posterior model for Gamma."""

    def __init__(
        self,
        n_sites: int,
        n_covariates: int,
        n_species: int,
        n_traits: int,
        hidden_units: tuple[int, ...] = (64, 64),
        min_scale: float = 1e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.n_traits = int(n_traits)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]
        self.head = DiagonalNormalGammaHead(
            n_covariates=self.n_covariates,
            n_traits=self.n_traits,
            min_scale=min_scale,
        )

    def call(self, inputs: dict[str, tf.Tensor]) -> GammaPosterior:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        traits = tf.cast(inputs["T"], tf.float32)
        _assert_trait_shape(
            design,
            response,
            traits,
            self.n_sites,
            self.n_covariates,
            self.n_species,
            self.n_traits,
        )

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(self.n_sites, tf.float32)
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(self.n_sites, tf.float32)
        beta_ridge = _ridge_beta_estimate(xtx, xty)
        gamma_ridge = _ridge_gamma_estimate(beta_ridge, traits)
        ttt = tf.einsum("bst,bsu->btu", traits, traits) / tf.cast(self.n_species, tf.float32)
        y_mean = tf.reduce_mean(response, axis=1)
        y_sd = tf.math.reduce_std(response, axis=1)
        trait_mean = tf.reduce_mean(traits, axis=1)
        trait_sd = tf.math.reduce_std(traits, axis=1)
        features = tf.concat(
            [
                tf.reshape(xty, (tf.shape(design)[0], -1)),
                tf.reshape(xtx, (tf.shape(design)[0], -1)),
                tf.reshape(beta_ridge, (tf.shape(design)[0], -1)),
                tf.reshape(gamma_ridge, (tf.shape(design)[0], -1)),
                tf.reshape(ttt, (tf.shape(design)[0], -1)),
                y_mean,
                y_sd,
                trait_mean,
                trait_sd,
            ],
            axis=-1,
        )
        for layer in self.encoder_layers:
            features = layer(features)
        residual = self.head(features)
        return GammaPosterior(mean=gamma_ridge + residual.mean, scale=residual.scale)


class IidLatentFactorPosteriorModel(tf.keras.Model):
    """Fixed-shape iid random-intercept latent-factor posterior model."""

    def __init__(
        self,
        n_sites: int,
        n_covariates: int,
        n_species: int,
        n_groups: int,
        n_factors: int = 1,
        beta_scale: float = 0.05,
        eta_scale: float = 0.05,
        lambda_scale: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.n_groups = int(n_groups)
        self.n_factors = int(n_factors)
        self.beta_scale = float(beta_scale)
        self.eta_scale = float(eta_scale)
        self.lambda_scale = float(lambda_scale)

    def call(self, inputs: dict[str, tf.Tensor]) -> IidLatentPosterior:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        group_codes = tf.cast(inputs["group_codes"], tf.int32)
        _assert_iid_latent_shape(
            design,
            response,
            group_codes,
            self.n_sites,
            self.n_covariates,
            self.n_species,
        )

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(self.n_sites, tf.float32)
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(self.n_sites, tf.float32)
        beta_ridge = _ridge_beta_estimate(xtx, xty)
        residual = response - tf.einsum("bnk,bks->bns", design, beta_ridge)
        group_residual = _group_means(residual, group_codes, self.n_groups)
        singular_values, left, right = tf.linalg.svd(group_residual, full_matrices=False)
        keep = min(self.n_factors, int(group_residual.shape[-1]), int(group_residual.shape[-2]))
        singular_values = singular_values[:, :keep]
        left = left[:, :, :keep]
        right = right[:, :, :keep]
        root = tf.sqrt(tf.maximum(singular_values, 0.0))
        eta_mean = left * root[:, None, :]
        lambda_mean = tf.transpose(right * root[:, None, :], [0, 2, 1])
        eta_mean = _pad_factor_axis(eta_mean, self.n_factors, axis=2)
        lambda_mean = _pad_factor_axis(lambda_mean, self.n_factors, axis=1)
        return IidLatentPosterior(
            beta_mean=beta_ridge,
            beta_scale=tf.ones_like(beta_ridge) * tf.cast(self.beta_scale, beta_ridge.dtype),
            eta_mean=eta_mean,
            eta_scale=tf.ones_like(eta_mean) * tf.cast(self.eta_scale, eta_mean.dtype),
            lambda_mean=lambda_mean,
            lambda_scale=tf.ones_like(lambda_mean) * tf.cast(self.lambda_scale, lambda_mean.dtype),
        )


class SpatialLatentFactorPosteriorModel(tf.keras.Model):
    """Full-spatial random-intercept latent-factor posterior prototype."""

    def __init__(
        self,
        n_sites: int,
        n_covariates: int,
        n_species: int,
        n_factors: int = 1,
        spatial_range: float = 0.25,
        beta_scale: float = 0.05,
        eta_scale: float = 0.05,
        lambda_scale: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.n_factors = int(n_factors)
        self.spatial_range = float(spatial_range)
        self.beta_scale = float(beta_scale)
        self.eta_scale = float(eta_scale)
        self.lambda_scale = float(lambda_scale)
        if self.spatial_range <= 0:
            raise ValueError("spatial_range must be positive")

    def call(self, inputs: dict[str, tf.Tensor]) -> IidLatentPosterior:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        coords = tf.cast(inputs["coords"], tf.float32)
        _assert_spatial_latent_shape(
            design,
            response,
            coords,
            self.n_sites,
            self.n_covariates,
            self.n_species,
        )

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(self.n_sites, tf.float32)
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(self.n_sites, tf.float32)
        beta_ridge = _ridge_beta_estimate(xtx, xty)
        residual = response - tf.einsum("bnk,bks->bns", design, beta_ridge)
        weights = _spatial_kernel(coords, self.spatial_range)
        smooth_residual = tf.einsum("bij,bjs->bis", weights, residual)
        singular_values, left, right = tf.linalg.svd(smooth_residual, full_matrices=False)
        keep = min(self.n_factors, int(smooth_residual.shape[-1]), int(smooth_residual.shape[-2]))
        singular_values = singular_values[:, :keep]
        left = left[:, :, :keep]
        right = right[:, :, :keep]
        root = tf.sqrt(tf.maximum(singular_values, 0.0))
        eta_mean = left * root[:, None, :]
        lambda_mean = tf.transpose(right * root[:, None, :], [0, 2, 1])
        eta_mean = _pad_factor_axis(eta_mean, self.n_factors, axis=2)
        lambda_mean = _pad_factor_axis(lambda_mean, self.n_factors, axis=1)
        return IidLatentPosterior(
            beta_mean=beta_ridge,
            beta_scale=tf.ones_like(beta_ridge) * tf.cast(self.beta_scale, beta_ridge.dtype),
            eta_mean=eta_mean,
            eta_scale=tf.ones_like(eta_mean) * tf.cast(self.eta_scale, eta_mean.dtype),
            lambda_mean=lambda_mean,
            lambda_scale=tf.ones_like(lambda_mean) * tf.cast(self.lambda_scale, lambda_mean.dtype),
        )


def _assert_fixed_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    n_sites: int,
    n_covariates: int,
    n_species: int,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3:
        raise ValueError("FixedShapeBetaPosteriorModel expects rank-3 X and Y tensors")
    if design.shape[1:] != (n_sites, n_covariates):
        raise ValueError(f"X must have trailing shape {(n_sites, n_covariates)}")
    if response.shape[1:] != (n_sites, n_species):
        raise ValueError(f"Y must have trailing shape {(n_sites, n_species)}")


def _assert_trait_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    traits: tf.Tensor,
    n_sites: int,
    n_covariates: int,
    n_species: int,
    n_traits: int,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3 or traits.shape.rank != 3:
        raise ValueError("TraitGammaPosteriorModel expects rank-3 X, Y, and T tensors")
    if design.shape[1:] != (n_sites, n_covariates):
        raise ValueError(f"X must have trailing shape {(n_sites, n_covariates)}")
    if response.shape[1:] != (n_sites, n_species):
        raise ValueError(f"Y must have trailing shape {(n_sites, n_species)}")
    if traits.shape[1:] != (n_species, n_traits):
        raise ValueError(f"T must have trailing shape {(n_species, n_traits)}")


def _assert_iid_latent_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    group_codes: tf.Tensor,
    n_sites: int,
    n_covariates: int,
    n_species: int,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3:
        raise ValueError("IidLatentFactorPosteriorModel expects rank-3 X and Y tensors")
    if group_codes.shape.rank != 2:
        raise ValueError("group_codes must have shape batch x sites")
    if design.shape[1:] != (n_sites, n_covariates):
        raise ValueError(f"X must have trailing shape {(n_sites, n_covariates)}")
    if response.shape[1:] != (n_sites, n_species):
        raise ValueError(f"Y must have trailing shape {(n_sites, n_species)}")
    if group_codes.shape[1:] != (n_sites,):
        raise ValueError(f"group_codes must have trailing shape {(n_sites,)}")


def _assert_spatial_latent_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    coords: tf.Tensor,
    n_sites: int,
    n_covariates: int,
    n_species: int,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3 or coords.shape.rank != 3:
        raise ValueError("SpatialLatentFactorPosteriorModel expects rank-3 X, Y, and coords tensors")
    if design.shape[1:] != (n_sites, n_covariates):
        raise ValueError(f"X must have trailing shape {(n_sites, n_covariates)}")
    if response.shape[1:] != (n_sites, n_species):
        raise ValueError(f"Y must have trailing shape {(n_sites, n_species)}")
    if coords.shape[1:] != (n_sites, 2):
        raise ValueError(f"coords must have trailing shape {(n_sites, 2)}")


def _assert_variable_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
    n_covariates: int,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3:
        raise ValueError("VariableShapeBetaPosteriorModel expects rank-3 X and Y tensors")
    if site_mask.shape.rank != 2 or species_mask.shape.rank != 2:
        raise ValueError("site_mask and species_mask must be rank-2 tensors")
    if design.shape[-1] != n_covariates:
        raise ValueError(f"X must have {n_covariates} covariates")
    if design.shape[1] != response.shape[1]:
        raise ValueError("X and Y must have the same padded site dimension")
    if response.shape[2] != species_mask.shape[1]:
        raise ValueError("Y and species_mask must have the same padded species dimension")
    if design.shape[1] != site_mask.shape[1]:
        raise ValueError("X and site_mask must have the same padded site dimension")


def _ridge_beta_estimate(xtx: tf.Tensor, xty: tf.Tensor, ridge: float = 1e-4) -> tf.Tensor:
    n_covariates = tf.shape(xtx)[-1]
    penalty = tf.eye(n_covariates, batch_shape=[tf.shape(xtx)[0]], dtype=xtx.dtype) * ridge
    return tf.linalg.solve(xtx + penalty, xty)


def _ridge_gamma_estimate(beta: tf.Tensor, traits: tf.Tensor, ridge: float = 1e-4) -> tf.Tensor:
    n_traits = tf.shape(traits)[-1]
    ttt = tf.einsum("bst,bsu->btu", traits, traits)
    penalty = tf.eye(n_traits, batch_shape=[tf.shape(traits)[0]], dtype=traits.dtype) * ridge
    beta_t = tf.einsum("bks,bst->bkt", beta, traits)
    solution = tf.linalg.solve(ttt + penalty, tf.transpose(beta_t, [0, 2, 1]))
    return tf.transpose(solution, [0, 2, 1])


def _group_means(values: tf.Tensor, group_codes: tf.Tensor, n_groups: int) -> tf.Tensor:
    one_hot = tf.one_hot(group_codes, depth=n_groups, dtype=values.dtype)
    totals = tf.einsum("bng,bns->bgs", one_hot, values)
    counts = tf.reduce_sum(one_hot, axis=1)
    return totals / tf.maximum(counts[:, :, None], 1.0)


def _pad_factor_axis(values: tf.Tensor, n_factors: int, axis: int) -> tf.Tensor:
    current = int(values.shape[axis])
    if current == n_factors:
        return values
    paddings = [[0, 0] for _ in range(values.shape.rank)]
    paddings[axis] = [0, n_factors - current]
    return tf.pad(values, paddings)


def _spatial_kernel(coords: tf.Tensor, spatial_range: float) -> tf.Tensor:
    delta = coords[:, :, None, :] - coords[:, None, :, :]
    distances = tf.sqrt(tf.reduce_sum(tf.square(delta), axis=-1) + 1e-12)
    weights = tf.exp(-distances / tf.cast(spatial_range, coords.dtype))
    return weights / tf.maximum(tf.reduce_sum(weights, axis=-1, keepdims=True), 1e-12)
