"""Experimental Neural-HMSC model prototypes."""

from __future__ import annotations

import tensorflow as tf

from pyhmsc.neural.posterior_heads import (
    BetaPosterior,
    DiagonalNormalBetaHead,
    DiagonalNormalGammaHead,
    GammaPosterior,
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
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]
        self.head = DiagonalNormalBetaHead(
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

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(self.n_sites, tf.float32)
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(self.n_sites, tf.float32)
        ridge = _ridge_beta_estimate(xtx, xty)
        y_mean = tf.reduce_mean(response, axis=1)
        y_sd = tf.math.reduce_std(response, axis=1)
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
        return BetaPosterior(mean=ridge + residual.mean, scale=residual.scale)


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
