"""Experimental Neural-HMSC model prototypes."""

from __future__ import annotations

import tensorflow as tf

from pyhmsc.neural.posterior_heads import BetaPosterior, DiagonalNormalBetaHead


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


def _ridge_beta_estimate(xtx: tf.Tensor, xty: tf.Tensor, ridge: float = 1e-4) -> tf.Tensor:
    n_covariates = tf.shape(xtx)[-1]
    penalty = tf.eye(n_covariates, batch_shape=[tf.shape(xtx)[0]], dtype=xtx.dtype) * ridge
    return tf.linalg.solve(xtx + penalty, xty)
