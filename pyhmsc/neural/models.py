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
        probit_anchor: str = "auto",
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        if distribution not in {"normal", "probit", "poisson"}:
            raise ValueError("distribution must be 'normal', 'probit', or 'poisson'")
        self.distribution = str(distribution)
        if probit_anchor == "auto":
            probit_anchor = "irls_laplace" if distribution == "probit" else "ridge"
        if probit_anchor not in {"ridge", "irls_laplace"}:
            raise ValueError("probit_anchor must be 'auto', 'ridge', or 'irls_laplace'")
        if probit_anchor == "irls_laplace" and distribution != "probit":
            raise ValueError(
                "irls_laplace probit anchor requires distribution='probit'"
            )
        if probit_anchor_iterations <= 0:
            raise ValueError("probit_anchor_iterations must be positive")
        if probit_anchor_prior_precision <= 0.0:
            raise ValueError("probit_anchor_prior_precision must be positive")
        if probit_anchor_eta_clip <= 0.0:
            raise ValueError("probit_anchor_eta_clip must be positive")
        self.probit_anchor = str(probit_anchor)
        self.probit_anchor_iterations = int(probit_anchor_iterations)
        self.probit_anchor_prior_precision = float(probit_anchor_prior_precision)
        self.probit_anchor_eta_clip = float(probit_anchor_eta_clip)
        if posterior_family not in {"diagonal_normal", "full_covariance_normal"}:
            raise ValueError(
                "posterior_family must be 'diagonal_normal' or 'full_covariance_normal'"
            )
        self.posterior_family = str(posterior_family)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu") for units in hidden_units
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

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor] | dict[str, tf.Tensor]
    ) -> BetaPosterior:
        if isinstance(inputs, dict):
            design = inputs["X"]
            response = inputs["Y"]
        else:
            design, response = inputs
        design = tf.cast(design, tf.float32)
        response = tf.cast(response, tf.float32)
        _assert_fixed_shape(
            design, response, self.n_sites, self.n_covariates, self.n_species
        )

        feature_response = (
            tf.math.log1p(response) if self.distribution == "poisson" else response
        )
        xty = tf.einsum("bnk,bns->bks", design, feature_response) / tf.cast(
            self.n_sites, tf.float32
        )
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(
            self.n_sites, tf.float32
        )
        ridge = _ridge_beta_estimate(xtx, xty)
        anchor = ridge
        anchor_scale = None
        if self.probit_anchor == "irls_laplace":
            anchor, anchor_scale = probit_irls_laplace_anchor(
                design,
                response,
                iterations=self.probit_anchor_iterations,
                prior_precision=self.probit_anchor_prior_precision,
                eta_clip=self.probit_anchor_eta_clip,
            )
        y_mean = tf.reduce_mean(feature_response, axis=1)
        y_sd = tf.math.reduce_std(feature_response, axis=1)
        feature_parts = [
            tf.reshape(xty, (tf.shape(design)[0], -1)),
            tf.reshape(xtx, (tf.shape(design)[0], -1)),
            tf.reshape(ridge, (tf.shape(design)[0], -1)),
        ]
        if anchor_scale is not None:
            feature_parts.extend(
                [
                    tf.reshape(anchor, (tf.shape(design)[0], -1)),
                    tf.reshape(
                        tf.math.log(tf.maximum(anchor_scale, 1e-6)),
                        (tf.shape(design)[0], -1),
                    ),
                ]
            )
        feature_parts.extend([y_mean, y_sd])
        features = tf.concat(feature_parts, axis=-1)
        for layer in self.encoder_layers:
            features = layer(features)
        residual = self.head(features)
        return BetaPosterior(
            mean=anchor + residual.mean,
            scale=residual.scale,
            scale_tril=residual.scale_tril,
        )


class VariableShapeBetaPosteriorModel(tf.keras.Model):
    """Masked variable-site/species fixed-effect posterior model."""

    def __init__(
        self,
        n_covariates: int = 3,
        hidden_units: tuple[int, ...] = (48, 48),
        min_scale: float = 1e-3,
        distribution: str = "normal",
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_covariates = int(n_covariates)
        self.hidden_units = tuple(int(value) for value in hidden_units)
        self.min_scale = float(min_scale)
        self.distribution = str(distribution).lower()
        self.probit_anchor_iterations = int(probit_anchor_iterations)
        self.probit_anchor_prior_precision = float(probit_anchor_prior_precision)
        self.probit_anchor_eta_clip = float(probit_anchor_eta_clip)
        if self.distribution not in {"normal", "probit"}:
            raise ValueError("variable-shape distribution must be 'normal' or 'probit'")
        if self.probit_anchor_iterations <= 0:
            raise ValueError("probit_anchor_iterations must be positive")
        if self.probit_anchor_prior_precision <= 0.0:
            raise ValueError("probit_anchor_prior_precision must be positive")
        if self.probit_anchor_eta_clip <= 0.0:
            raise ValueError("probit_anchor_eta_clip must be positive")
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in self.hidden_units
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
        _assert_variable_shape(
            design, response, site_mask, species_mask, self.n_covariates
        )

        site_weights = site_mask[:, :, None]
        masked_design = design * site_weights
        masked_response = response * site_weights * species_mask[:, None, :]
        site_count = tf.maximum(tf.reduce_sum(site_mask, axis=1), 1.0)
        xty = (
            tf.einsum("bnk,bns->bks", masked_design, masked_response)
            / site_count[:, None, None]
        )
        xtx = (
            tf.einsum("bnk,bnl->bkl", masked_design, masked_design)
            / site_count[:, None, None]
        )
        ridge = _ridge_beta_estimate(xtx, xty)
        if self.distribution == "probit":
            anchor, anchor_scale = probit_irls_laplace_anchor(
                design,
                response,
                iterations=self.probit_anchor_iterations,
                prior_precision=self.probit_anchor_prior_precision,
                eta_clip=self.probit_anchor_eta_clip,
                site_mask=site_mask,
            )
        else:
            anchor = ridge
            anchor_scale = None
        y_mean = tf.reduce_sum(masked_response, axis=1) / site_count[:, None]
        centered = (
            (response - y_mean[:, None, :]) * site_weights * species_mask[:, None, :]
        )
        y_sd = tf.sqrt(
            tf.reduce_sum(tf.square(centered), axis=1) / site_count[:, None] + 1e-8
        )
        xtx_features = tf.tile(
            tf.reshape(
                xtx, (tf.shape(design)[0], 1, self.n_covariates * self.n_covariates)
            ),
            [1, tf.shape(response)[2], 1],
        )
        species_features = tf.concat(
            [
                tf.transpose(xty, [0, 2, 1]),
                tf.transpose(anchor, [0, 2, 1]),
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
        if anchor_scale is None:
            residual_scale = tf.transpose(
                tf.nn.softplus(scale_raw) + self.min_scale, [0, 2, 1]
            )
        else:
            log_scale_adjustment = tf.transpose(
                tf.clip_by_value(scale_raw, -4.0, 4.0), [0, 2, 1]
            )
            residual_scale = tf.maximum(
                anchor_scale * tf.exp(log_scale_adjustment), self.min_scale
            )
        mask = species_mask[:, None, :]
        return BetaPosterior(
            mean=(anchor + residual_mean) * mask,
            scale=residual_scale * mask,
        )


class VariableDesignBetaPosteriorModel(tf.keras.Model):
    """Masked fixed-effect posterior model with a shared coefficient head."""

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (48, 48),
        min_scale: float = 1e-3,
        mean_correction_limit: float = 0.5,
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = tuple(int(value) for value in hidden_units)
        self.min_scale = float(min_scale)
        self.mean_correction_limit = float(mean_correction_limit)
        self.probit_anchor_iterations = int(probit_anchor_iterations)
        self.probit_anchor_prior_precision = float(probit_anchor_prior_precision)
        self.probit_anchor_eta_clip = float(probit_anchor_eta_clip)
        if self.min_scale <= 0.0 or self.mean_correction_limit <= 0.0:
            raise ValueError("scale and mean correction limits must be positive")
        if self.probit_anchor_iterations <= 0:
            raise ValueError("probit_anchor_iterations must be positive")
        if self.probit_anchor_prior_precision <= 0.0:
            raise ValueError("probit_anchor_prior_precision must be positive")
        if self.probit_anchor_eta_clip <= 0.0:
            raise ValueError("probit_anchor_eta_clip must be positive")
        self.coefficient_encoder = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in self.hidden_units
        ]
        self.shared_projection = tf.keras.layers.Dense(
            2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, inputs: dict[str, tf.Tensor]) -> BetaPosterior:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        site_mask = tf.cast(inputs["site_mask"], tf.float32)
        species_mask = tf.cast(inputs["species_mask"], tf.float32)
        covariate_mask = tf.cast(inputs["covariate_mask"], tf.float32)
        _assert_variable_design_shape(
            design, response, site_mask, species_mask, covariate_mask
        )

        site_weights = site_mask[:, :, None]
        species_weights = species_mask[:, None, :]
        covariate_weights = covariate_mask[:, None, :]
        masked_design = design * site_weights * covariate_weights
        masked_response = response * site_weights * species_weights
        site_count = tf.maximum(tf.reduce_sum(site_mask, axis=1), 1.0)
        species_count = tf.maximum(tf.reduce_sum(species_mask, axis=1), 1.0)
        covariate_count = tf.maximum(tf.reduce_sum(covariate_mask, axis=1), 1.0)

        anchor, anchor_scale = probit_irls_laplace_anchor(
            masked_design,
            masked_response,
            iterations=self.probit_anchor_iterations,
            prior_precision=self.probit_anchor_prior_precision,
            eta_clip=self.probit_anchor_eta_clip,
            site_mask=site_mask,
        )
        xty = (
            tf.einsum("bnk,bns->bks", masked_design, masked_response)
            / site_count[:, None, None]
        )
        xtx = (
            tf.einsum("bnk,bnl->bkl", masked_design, masked_design)
            / site_count[:, None, None]
        )
        design_mean = tf.reduce_sum(masked_design, axis=1) / site_count[:, None]
        design_second = (
            tf.reduce_sum(tf.square(masked_design), axis=1) / site_count[:, None]
        )
        design_sd = tf.sqrt(
            tf.maximum(design_second - tf.square(design_mean), 0.0) + 1e-8
        )
        design_information = tf.linalg.diag_part(xtx)

        n_covariates = tf.shape(design)[2]
        pair_mask = covariate_mask[:, :, None] * covariate_mask[:, None, :]
        off_diagonal = 1.0 - tf.eye(
            n_covariates, batch_shape=[tf.shape(design)[0]], dtype=design.dtype
        )
        cross = xtx * pair_mask * off_diagonal
        cross_denominator = tf.maximum(covariate_count - 1.0, 1.0)[:, None]
        cross_abs_mean = tf.reduce_sum(tf.abs(cross), axis=2) / cross_denominator
        cross_rms = tf.sqrt(
            tf.reduce_sum(tf.square(cross), axis=2) / cross_denominator + 1e-8
        )

        y_mean = tf.reduce_sum(masked_response, axis=1) / site_count[:, None]
        centered_response = (
            (response - y_mean[:, None, :]) * site_weights * species_weights
        )
        y_sd = tf.sqrt(
            tf.reduce_sum(tf.square(centered_response), axis=1) / site_count[:, None]
            + 1e-8
        )

        target_shape = tf.stack(
            [tf.shape(design)[0], n_covariates, tf.shape(response)[2]]
        )

        def coefficient_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, :, None], target_shape)[..., None]

        def species_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, None, :], target_shape)[..., None]

        def global_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, None, None], target_shape)[..., None]

        intercept = tf.one_hot(0, n_covariates, dtype=design.dtype)[None, :]
        intercept = intercept * covariate_mask
        features = tf.concat(
            [
                xty[..., None],
                anchor[..., None],
                anchor_scale[..., None],
                coefficient_feature(design_mean),
                coefficient_feature(design_sd),
                coefficient_feature(design_information),
                coefficient_feature(cross_abs_mean),
                coefficient_feature(cross_rms),
                coefficient_feature(intercept),
                coefficient_feature(covariate_mask),
                species_feature(y_mean),
                species_feature(y_sd),
                species_feature(species_mask),
                global_feature(tf.math.log1p(site_count)),
                global_feature(tf.math.log1p(species_count)),
                global_feature(tf.math.log1p(covariate_count)),
            ],
            axis=-1,
        )
        encoded = features
        for layer in self.coefficient_encoder:
            encoded = layer(encoded)
        raw = self.shared_projection(encoded)
        mean_adjustment = self.mean_correction_limit * tf.tanh(raw[..., 0])
        log_scale_adjustment = tf.clip_by_value(raw[..., 1], -4.0, 4.0)
        active_mask = covariate_mask[:, :, None] * species_mask[:, None, :]
        return BetaPosterior(
            mean=(anchor + mean_adjustment) * active_mask,
            scale=(
                tf.maximum(anchor_scale * tf.exp(log_scale_adjustment), self.min_scale)
                * active_mask
            ),
        )


class GatedVariableDesignBetaPosteriorModel(tf.keras.Model):
    """Variable-design posterior with learned anchor-residual support gating."""

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (48, 48),
        min_scale: float = 1e-3,
        mean_correction_limit: float = 0.5,
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        min_support_ratio: float = 1.5,
        max_support_ratio: float = 64.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = tuple(int(value) for value in hidden_units)
        self.min_scale = float(min_scale)
        self.mean_correction_limit = float(mean_correction_limit)
        self.probit_anchor_iterations = int(probit_anchor_iterations)
        self.probit_anchor_prior_precision = float(probit_anchor_prior_precision)
        self.probit_anchor_eta_clip = float(probit_anchor_eta_clip)
        self.min_support_ratio = float(min_support_ratio)
        self.max_support_ratio = float(max_support_ratio)
        if self.min_scale <= 0.0 or self.mean_correction_limit <= 0.0:
            raise ValueError("scale and mean correction limits must be positive")
        if self.probit_anchor_iterations <= 0:
            raise ValueError("probit_anchor_iterations must be positive")
        if self.probit_anchor_prior_precision <= 0.0:
            raise ValueError("probit_anchor_prior_precision must be positive")
        if self.probit_anchor_eta_clip <= 0.0:
            raise ValueError("probit_anchor_eta_clip must be positive")
        if not 0.0 < self.min_support_ratio < self.max_support_ratio:
            raise ValueError("support-ratio bounds must be positive and ordered")
        self.coefficient_encoder = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in self.hidden_units
        ]
        self.shared_projection = tf.keras.layers.Dense(
            3,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, inputs: dict[str, tf.Tensor]) -> BetaPosterior:
        anchor, anchor_scale, raw, active_mask = self._components(inputs)
        mean_adjustment = self.mean_correction_limit * tf.tanh(raw[..., 0])
        log_scale_adjustment = tf.clip_by_value(raw[..., 1], -4.0, 4.0)
        support_gate = tf.math.sigmoid(raw[..., 2])
        residual_mean = anchor + mean_adjustment
        posterior_mean = (
            (1.0 - support_gate) * anchor + support_gate * residual_mean
        )
        return BetaPosterior(
            mean=posterior_mean * active_mask,
            scale=(
                tf.maximum(anchor_scale * tf.exp(log_scale_adjustment), self.min_scale)
                * active_mask
            ),
        )

    def support_gate(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        """Return the active coefficient/species support mixture weights."""
        _, _, raw, active_mask = self._components(inputs)
        return tf.math.sigmoid(raw[..., 2]) * active_mask

    def _components(
        self, inputs: dict[str, tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        site_mask = tf.cast(inputs["site_mask"], tf.float32)
        species_mask = tf.cast(inputs["species_mask"], tf.float32)
        covariate_mask = tf.cast(inputs["covariate_mask"], tf.float32)
        _assert_variable_design_shape(
            design, response, site_mask, species_mask, covariate_mask
        )

        site_weights = site_mask[:, :, None]
        species_weights = species_mask[:, None, :]
        covariate_weights = covariate_mask[:, None, :]
        masked_design = design * site_weights * covariate_weights
        masked_response = response * site_weights * species_weights
        site_count = tf.maximum(tf.reduce_sum(site_mask, axis=1), 1.0)
        species_count = tf.maximum(tf.reduce_sum(species_mask, axis=1), 1.0)
        covariate_count = tf.maximum(tf.reduce_sum(covariate_mask, axis=1), 1.0)

        anchor, anchor_scale = probit_irls_laplace_anchor(
            masked_design,
            masked_response,
            iterations=self.probit_anchor_iterations,
            prior_precision=self.probit_anchor_prior_precision,
            eta_clip=self.probit_anchor_eta_clip,
            site_mask=site_mask,
        )
        xty = (
            tf.einsum("bnk,bns->bks", masked_design, masked_response)
            / site_count[:, None, None]
        )
        xtx = (
            tf.einsum("bnk,bnl->bkl", masked_design, masked_design)
            / site_count[:, None, None]
        )
        design_mean = tf.reduce_sum(masked_design, axis=1) / site_count[:, None]
        design_second = (
            tf.reduce_sum(tf.square(masked_design), axis=1) / site_count[:, None]
        )
        design_sd = tf.sqrt(
            tf.maximum(design_second - tf.square(design_mean), 0.0) + 1e-8
        )
        design_information = tf.linalg.diag_part(xtx)
        normalized_design_information = (
            design_information / covariate_count[:, None]
        )

        n_covariates = tf.shape(design)[2]
        pair_mask = covariate_mask[:, :, None] * covariate_mask[:, None, :]
        off_diagonal = 1.0 - tf.eye(
            n_covariates, batch_shape=[tf.shape(design)[0]], dtype=design.dtype
        )
        cross = xtx * pair_mask * off_diagonal
        cross_denominator = tf.maximum(covariate_count - 1.0, 1.0)[:, None]
        cross_abs_mean = tf.reduce_sum(tf.abs(cross), axis=2) / cross_denominator
        cross_rms = tf.sqrt(
            tf.reduce_sum(tf.square(cross), axis=2) / cross_denominator + 1e-8
        )

        y_mean = tf.reduce_sum(masked_response, axis=1) / site_count[:, None]
        centered_response = (
            (response - y_mean[:, None, :]) * site_weights * species_weights
        )
        y_sd = tf.sqrt(
            tf.reduce_sum(tf.square(centered_response), axis=1) / site_count[:, None]
            + 1e-8
        )
        support_ratio = tf.clip_by_value(
            site_count / covariate_count,
            self.min_support_ratio,
            self.max_support_ratio,
        )

        target_shape = tf.stack(
            [tf.shape(design)[0], n_covariates, tf.shape(response)[2]]
        )

        def coefficient_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, :, None], target_shape)[..., None]

        def species_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, None, :], target_shape)[..., None]

        def global_feature(value: tf.Tensor) -> tf.Tensor:
            return tf.broadcast_to(value[:, None, None], target_shape)[..., None]

        intercept = tf.one_hot(0, n_covariates, dtype=design.dtype)[None, :]
        intercept = intercept * covariate_mask
        features = tf.concat(
            [
                xty[..., None],
                anchor[..., None],
                anchor_scale[..., None],
                coefficient_feature(design_mean),
                coefficient_feature(design_sd),
                coefficient_feature(design_information),
                coefficient_feature(normalized_design_information),
                coefficient_feature(cross_abs_mean),
                coefficient_feature(cross_rms),
                coefficient_feature(intercept),
                coefficient_feature(covariate_mask),
                species_feature(y_mean),
                species_feature(y_sd),
                species_feature(species_mask),
                global_feature(tf.math.log1p(site_count)),
                global_feature(tf.math.log1p(species_count)),
                global_feature(tf.math.log1p(covariate_count)),
                global_feature(tf.math.log1p(site_count / covariate_count)),
                global_feature(support_ratio),
            ],
            axis=-1,
        )
        encoded = features
        for layer in self.coefficient_encoder:
            encoded = layer(encoded)
        raw = self.shared_projection(encoded)
        active_mask = covariate_mask[:, :, None] * species_mask[:, None, :]
        return anchor, anchor_scale, raw, active_mask


class TraitGammaPosteriorModel(tf.keras.Model):
    """Fixed-shape trait-mediated posterior model for Gamma.

    The statistical anchor first estimates species-specific ``Beta`` and then
    projects it onto the compiler-emitted trait design with inverse-variance
    weights.  The neural component is deliberately a bounded correction around
    that anchor rather than a replacement for it.
    """

    def __init__(
        self,
        n_sites: int,
        n_covariates: int,
        n_species: int,
        n_traits: int,
        hidden_units: tuple[int, ...] = (64, 64),
        min_scale: float = 1e-3,
        distribution: str = "normal",
        probit_anchor_iterations: int = 8,
        probit_anchor_prior_precision: float = 1.0,
        probit_anchor_eta_clip: float = 6.0,
        gamma_prior_precision: float = 1.0,
        max_mean_adjustment: float = 0.5,
        max_log_scale_adjustment: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.n_sites = int(n_sites)
        self.n_covariates = int(n_covariates)
        self.n_species = int(n_species)
        self.n_traits = int(n_traits)
        self.hidden_units = tuple(int(value) for value in hidden_units)
        self.min_scale = float(min_scale)
        self.distribution = str(distribution).lower()
        if self.distribution == "gaussian":
            self.distribution = "normal"
        if self.distribution not in {"normal", "probit"}:
            raise ValueError("TraitGammaPosteriorModel supports normal or probit")
        self.probit_anchor_iterations = int(probit_anchor_iterations)
        self.probit_anchor_prior_precision = float(probit_anchor_prior_precision)
        self.probit_anchor_eta_clip = float(probit_anchor_eta_clip)
        self.gamma_prior_precision = float(gamma_prior_precision)
        self.max_mean_adjustment = float(max_mean_adjustment)
        self.max_log_scale_adjustment = float(max_log_scale_adjustment)
        self.encoder_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in self.hidden_units
        ]
        output_size = self.n_covariates * self.n_traits
        self.mean_adjustment = tf.keras.layers.Dense(
            output_size,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )
        self.log_scale_adjustment = tf.keras.layers.Dense(
            output_size,
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def beta_anchor(self, inputs: dict[str, tf.Tensor]) -> BetaPosterior:
        """Return the species-specific statistical anchor used by Gamma."""
        design = tf.cast(inputs["X"], tf.float32)
        response = tf.cast(inputs["Y"], tf.float32)
        if self.distribution == "probit":
            mean, scale = probit_irls_laplace_anchor(
                design,
                response,
                iterations=self.probit_anchor_iterations,
                prior_precision=self.probit_anchor_prior_precision,
                eta_clip=self.probit_anchor_eta_clip,
            )
        else:
            mean, scale = _normal_beta_laplace_anchor(
                design,
                response,
                prior_precision=self.probit_anchor_prior_precision,
            )
        return BetaPosterior(mean=mean, scale=scale)

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

        beta_anchor = self.beta_anchor({"X": design, "Y": response})
        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(
            self.n_sites, tf.float32
        )
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(
            self.n_sites, tf.float32
        )
        gamma_anchor, gamma_scale = _joint_trait_gamma_anchor(
            design,
            response,
            traits,
            distribution=self.distribution,
            probit_iterations=self.probit_anchor_iterations,
            prior_precision=self.gamma_prior_precision,
            eta_clip=self.probit_anchor_eta_clip,
            min_scale=self.min_scale,
        )
        ttt = tf.einsum("bst,bsu->btu", traits, traits) / tf.cast(
            self.n_species, tf.float32
        )
        y_species_mean = tf.reduce_mean(response, axis=1)
        trait_mean = tf.reduce_mean(traits, axis=1)
        trait_sd = tf.math.reduce_std(traits, axis=1)
        features = tf.concat(
            [
                tf.reshape(xtx, (tf.shape(design)[0], -1)),
                tf.reduce_mean(xty, axis=2),
                tf.math.reduce_std(xty, axis=2),
                tf.reduce_mean(beta_anchor.mean, axis=2),
                tf.math.reduce_std(beta_anchor.mean, axis=2),
                tf.reduce_mean(beta_anchor.scale, axis=2),
                tf.math.reduce_std(beta_anchor.scale, axis=2),
                tf.reshape(gamma_anchor, (tf.shape(design)[0], -1)),
                tf.reshape(gamma_scale, (tf.shape(design)[0], -1)),
                tf.reshape(ttt, (tf.shape(design)[0], -1)),
                tf.reduce_mean(y_species_mean, axis=1, keepdims=True),
                tf.math.reduce_std(y_species_mean, axis=1, keepdims=True),
                trait_mean,
                trait_sd,
            ],
            axis=-1,
        )
        for layer in self.encoder_layers:
            features = layer(features)
        batch = tf.shape(design)[0]
        mean_raw = tf.reshape(
            self.mean_adjustment(features),
            (batch, self.n_covariates, self.n_traits),
        )
        log_scale_raw = tf.reshape(
            self.log_scale_adjustment(features),
            (batch, self.n_covariates, self.n_traits),
        )
        mean = gamma_anchor + (
            self.max_mean_adjustment * gamma_scale * tf.tanh(mean_raw)
        )
        log_scale = tf.clip_by_value(
            log_scale_raw,
            -self.max_log_scale_adjustment,
            self.max_log_scale_adjustment,
        )
        scale = tf.maximum(gamma_scale * tf.exp(log_scale), self.min_scale)
        return GammaPosterior(mean=mean, scale=scale)


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

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(
            self.n_sites, tf.float32
        )
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(
            self.n_sites, tf.float32
        )
        beta_ridge = _ridge_beta_estimate(xtx, xty)
        residual = response - tf.einsum("bnk,bks->bns", design, beta_ridge)
        group_residual = _group_means(residual, group_codes, self.n_groups)
        singular_values, left, right = tf.linalg.svd(
            group_residual, full_matrices=False
        )
        keep = min(
            self.n_factors, int(group_residual.shape[-1]), int(group_residual.shape[-2])
        )
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
            beta_scale=tf.ones_like(beta_ridge)
            * tf.cast(self.beta_scale, beta_ridge.dtype),
            eta_mean=eta_mean,
            eta_scale=tf.ones_like(eta_mean) * tf.cast(self.eta_scale, eta_mean.dtype),
            lambda_mean=lambda_mean,
            lambda_scale=tf.ones_like(lambda_mean)
            * tf.cast(self.lambda_scale, lambda_mean.dtype),
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

        xty = tf.einsum("bnk,bns->bks", design, response) / tf.cast(
            self.n_sites, tf.float32
        )
        xtx = tf.einsum("bnk,bnl->bkl", design, design) / tf.cast(
            self.n_sites, tf.float32
        )
        beta_ridge = _ridge_beta_estimate(xtx, xty)
        residual = response - tf.einsum("bnk,bks->bns", design, beta_ridge)
        weights = _spatial_kernel(coords, self.spatial_range)
        smooth_residual = tf.einsum("bij,bjs->bis", weights, residual)
        singular_values, left, right = tf.linalg.svd(
            smooth_residual, full_matrices=False
        )
        keep = min(
            self.n_factors,
            int(smooth_residual.shape[-1]),
            int(smooth_residual.shape[-2]),
        )
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
            beta_scale=tf.ones_like(beta_ridge)
            * tf.cast(self.beta_scale, beta_ridge.dtype),
            eta_mean=eta_mean,
            eta_scale=tf.ones_like(eta_mean) * tf.cast(self.eta_scale, eta_mean.dtype),
            lambda_mean=lambda_mean,
            lambda_scale=tf.ones_like(lambda_mean)
            * tf.cast(self.lambda_scale, lambda_mean.dtype),
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
        raise ValueError(
            "SpatialLatentFactorPosteriorModel expects rank-3 X, Y, and coords tensors"
        )
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
        raise ValueError(
            "VariableShapeBetaPosteriorModel expects rank-3 X and Y tensors"
        )
    if site_mask.shape.rank != 2 or species_mask.shape.rank != 2:
        raise ValueError("site_mask and species_mask must be rank-2 tensors")
    if design.shape[-1] != n_covariates:
        raise ValueError(f"X must have {n_covariates} covariates")
    if design.shape[1] != response.shape[1]:
        raise ValueError("X and Y must have the same padded site dimension")
    if response.shape[2] != species_mask.shape[1]:
        raise ValueError(
            "Y and species_mask must have the same padded species dimension"
        )
    if design.shape[1] != site_mask.shape[1]:
        raise ValueError("X and site_mask must have the same padded site dimension")


def _assert_variable_design_shape(
    design: tf.Tensor,
    response: tf.Tensor,
    site_mask: tf.Tensor,
    species_mask: tf.Tensor,
    covariate_mask: tf.Tensor,
) -> None:
    if design.shape.rank != 3 or response.shape.rank != 3:
        raise ValueError("variable-design X and Y tensors must be rank 3")
    if site_mask.shape.rank != 2:
        raise ValueError("variable-design site_mask must be rank 2")
    if species_mask.shape.rank != 2 or covariate_mask.shape.rank != 2:
        raise ValueError("variable-design species/covariate masks must be rank 2")
    tf.debugging.assert_equal(
        tf.shape(design)[:2],
        tf.shape(response)[:2],
        message="variable-design X and Y batch/site dimensions differ",
    )
    tf.debugging.assert_equal(
        tf.shape(design)[:2],
        tf.shape(site_mask),
        message="variable-design X and site_mask dimensions differ",
    )
    tf.debugging.assert_equal(
        tf.stack([tf.shape(response)[0], tf.shape(response)[2]]),
        tf.shape(species_mask),
        message="variable-design Y and species_mask dimensions differ",
    )
    tf.debugging.assert_equal(
        tf.stack([tf.shape(design)[0], tf.shape(design)[2]]),
        tf.shape(covariate_mask),
        message="variable-design X and covariate_mask dimensions differ",
    )


def _ridge_beta_estimate(
    xtx: tf.Tensor, xty: tf.Tensor, ridge: float = 1e-4
) -> tf.Tensor:
    n_covariates = tf.shape(xtx)[-1]
    penalty = (
        tf.eye(n_covariates, batch_shape=[tf.shape(xtx)[0]], dtype=xtx.dtype) * ridge
    )
    return tf.linalg.solve(xtx + penalty, xty)


def probit_irls_laplace_anchor(
    design: tf.Tensor,
    response: tf.Tensor,
    *,
    iterations: int = 8,
    prior_precision: float = 1.0,
    eta_clip: float = 6.0,
    site_mask: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a penalized probit mode and Laplace marginal standard deviations."""
    posterior = probit_irls_laplace_full_anchor(
        design,
        response,
        iterations=iterations,
        prior_precision=prior_precision,
        eta_clip=eta_clip,
        site_mask=site_mask,
    )
    return posterior.mean, posterior.scale


def probit_irls_laplace_full_anchor(
    design: tf.Tensor,
    response: tf.Tensor,
    *,
    iterations: int = 8,
    prior_precision: float = 1.0,
    eta_clip: float = 6.0,
    site_mask: tf.Tensor | None = None,
) -> BetaPosterior:
    """Return the penalized probit mode and full per-species Laplace covariance."""
    design = tf.cast(design, tf.float32)
    response = tf.cast(response, tf.float32)
    if design.shape.rank != 3 or response.shape.rank != 3:
        raise ValueError("probit anchor expects batch x site x variable tensors")
    if design.shape[0] != response.shape[0] or design.shape[1] != response.shape[1]:
        raise ValueError("probit anchor X and Y batch/site dimensions must match")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if prior_precision <= 0.0 or eta_clip <= 0.0:
        raise ValueError("prior_precision and eta_clip must be positive")

    if site_mask is None:
        site_mask = tf.ones(tf.shape(design)[:2], dtype=design.dtype)
    else:
        site_mask = tf.cast(site_mask, design.dtype)
        if site_mask.shape.rank != 2 or site_mask.shape != design.shape[:2]:
            raise ValueError("probit anchor site_mask must match batch x site")
    site_weights = site_mask[:, :, None]
    masked_design = design * site_weights
    masked_response = response * site_weights
    site_count = tf.maximum(tf.reduce_sum(site_mask, axis=1), 1.0)

    n_covariates = tf.shape(design)[2]
    xtx = tf.einsum("bnk,bnl->bkl", masked_design, masked_design)
    xty = tf.einsum("bnk,bns->bks", masked_design, masked_response)
    beta = _ridge_beta_estimate(xtx, xty, ridge=prior_precision)
    prevalence = tf.clip_by_value(
        tf.reduce_sum(masked_response, axis=1) / site_count[:, None],
        1e-4,
        1.0 - 1e-4,
    )
    intercept = tf.sqrt(tf.constant(2.0, dtype=design.dtype)) * tf.math.erfinv(
        2.0 * prevalence - 1.0
    )
    beta = tf.concat([intercept[:, None, :], beta[:, 1:, :]], axis=1)

    precision = None
    for _ in range(int(iterations)):
        eta = tf.clip_by_value(
            tf.einsum("bnk,bks->bns", design, beta),
            -float(eta_clip),
            float(eta_clip),
        )
        probability = tf.clip_by_value(
            0.5
            * (1.0 + tf.math.erf(eta / tf.sqrt(tf.constant(2.0, dtype=design.dtype)))),
            1e-6,
            1.0 - 1e-6,
        )
        density = tf.maximum(
            tf.exp(-0.5 * tf.square(eta))
            / tf.sqrt(tf.constant(2.0 * 3.141592653589793, dtype=design.dtype)),
            1e-6,
        )
        weight = tf.square(density) / (probability * (1.0 - probability)) * site_weights
        working_response = eta + (response - probability) / density
        precision = tf.einsum("bnk,bns,bnl->bskl", design, weight, design)
        penalty = tf.eye(
            n_covariates,
            batch_shape=[tf.shape(design)[0], tf.shape(response)[2]],
            dtype=design.dtype,
        ) * float(prior_precision)
        precision = precision + penalty
        right_hand_side = tf.einsum(
            "bnk,bns,bns->bsk", design, weight, working_response
        )
        beta_by_species = tf.linalg.solve(precision, right_hand_side[..., None])
        beta = tf.transpose(beta_by_species[..., 0], [0, 2, 1])

    if precision is None:
        raise RuntimeError("probit anchor did not execute an IRLS iteration")
    covariance = tf.linalg.inv(precision)
    marginal_scale = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariance), 1e-12))
    scale_tril = tf.linalg.cholesky(covariance)
    return BetaPosterior(
        mean=tf.stop_gradient(beta),
        scale=tf.stop_gradient(tf.transpose(marginal_scale, [0, 2, 1])),
        scale_tril=tf.stop_gradient(scale_tril),
    )


def _ridge_gamma_estimate(
    beta: tf.Tensor, traits: tf.Tensor, ridge: float = 1e-4
) -> tf.Tensor:
    n_traits = tf.shape(traits)[-1]
    ttt = tf.einsum("bst,bsu->btu", traits, traits)
    penalty = (
        tf.eye(n_traits, batch_shape=[tf.shape(traits)[0]], dtype=traits.dtype) * ridge
    )
    beta_t = tf.einsum("bks,bst->bkt", beta, traits)
    solution = tf.linalg.solve(ttt + penalty, tf.transpose(beta_t, [0, 2, 1]))
    return tf.transpose(solution, [0, 2, 1])


def _normal_beta_laplace_anchor(
    design: tf.Tensor,
    response: tf.Tensor,
    *,
    prior_precision: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ridge Gaussian Beta means and diagonal posterior scales."""
    xtx = tf.einsum("bnk,bnl->bkl", design, design)
    xty = tf.einsum("bnk,bns->bks", design, response)
    n_covariates = tf.shape(design)[2]
    penalty = tf.eye(
        n_covariates,
        batch_shape=[tf.shape(design)[0]],
        dtype=design.dtype,
    ) * float(prior_precision)
    precision = xtx + penalty
    mean = tf.linalg.solve(precision, xty)
    residual = response - tf.einsum("bnk,bks->bns", design, mean)
    degrees = tf.maximum(
        tf.cast(tf.shape(design)[1] - tf.shape(design)[2], design.dtype),
        1.0,
    )
    variance = tf.reduce_sum(tf.square(residual), axis=1) / degrees
    covariance_diagonal = tf.linalg.diag_part(tf.linalg.inv(precision))
    scale = tf.sqrt(
        tf.maximum(covariance_diagonal[:, :, None] * variance[:, None, :], 1e-12)
    )
    return tf.stop_gradient(mean), tf.stop_gradient(scale)


def _weighted_gamma_anchor(
    beta_mean: tf.Tensor,
    beta_scale: tf.Tensor,
    traits: tf.Tensor,
    *,
    prior_precision: float,
    min_scale: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Project Beta onto traits using its marginal posterior precision."""
    weights = tf.math.reciprocal(tf.maximum(tf.square(beta_scale), 1e-8))
    precision = tf.einsum("bst,bks,bsu->bktu", traits, weights, traits)
    n_traits = tf.shape(traits)[2]
    penalty = tf.eye(
        n_traits,
        batch_shape=[tf.shape(traits)[0], tf.shape(beta_mean)[1]],
        dtype=traits.dtype,
    ) * float(prior_precision)
    precision = precision + penalty
    right_hand_side = tf.einsum("bst,bks,bks->bkt", traits, weights, beta_mean)
    mean = tf.linalg.solve(precision, right_hand_side[..., None])[..., 0]
    covariance = tf.linalg.inv(precision)
    scale = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariance), min_scale**2))
    return tf.stop_gradient(mean), tf.stop_gradient(scale)


def _joint_trait_gamma_anchor(
    design: tf.Tensor,
    response: tf.Tensor,
    traits: tf.Tensor,
    *,
    distribution: str,
    probit_iterations: int,
    prior_precision: float,
    eta_clip: float,
    min_scale: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Estimate Gamma directly from the joint site-species likelihood."""
    batch = tf.shape(design)[0]
    n_covariates = tf.shape(design)[2]
    n_traits = tf.shape(traits)[2]
    joint_design = tf.einsum("bnc,bst->bnsct", design, traits)
    joint_design = tf.reshape(
        joint_design,
        (batch, tf.shape(design)[1] * tf.shape(traits)[1], n_covariates * n_traits),
    )
    joint_response = tf.reshape(
        response,
        (batch, tf.shape(design)[1] * tf.shape(traits)[1], 1),
    )
    if distribution == "probit":
        mean, scale = _bounded_probit_irls_laplace_anchor(
            joint_design,
            joint_response,
            iterations=probit_iterations,
            prior_precision=prior_precision,
            eta_clip=eta_clip,
        )
    else:
        mean, scale = _normal_beta_laplace_anchor(
            joint_design,
            joint_response,
            prior_precision=prior_precision,
        )
    mean = tf.reshape(mean[..., 0], (batch, n_covariates, n_traits))
    scale = tf.reshape(scale[..., 0], (batch, n_covariates, n_traits))
    return tf.stop_gradient(mean), tf.maximum(tf.stop_gradient(scale), min_scale)


def _bounded_probit_irls_laplace_anchor(
    design: tf.Tensor,
    response: tf.Tensor,
    *,
    iterations: int,
    prior_precision: float,
    eta_clip: float,
    coefficient_clip: float = 4.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Numerically bounded probit IRLS for aggregate trait regressions.

    Aggregate ``X (x) T`` designs can be nearly separated even when every
    species-level regression is regular.  Clipping the working response and
    mode prevents the tail-density approximation from creating unbounded
    Newton steps while the Gaussian prior still controls the final precision.
    """
    design = tf.cast(design, tf.float32)
    response = tf.cast(response, tf.float32)
    xtx = tf.einsum("bnk,bnl->bkl", design, design)
    xty = tf.einsum("bnk,bns->bks", design, response)
    n_covariates = tf.shape(design)[2]
    beta = _ridge_beta_estimate(xtx, xty, ridge=prior_precision)
    precision = None
    for _ in range(int(iterations)):
        eta = tf.clip_by_value(
            tf.einsum("bnk,bks->bns", design, beta), -eta_clip, eta_clip
        )
        probability = tf.clip_by_value(
            0.5 * (1.0 + tf.math.erf(eta / tf.sqrt(tf.constant(2.0, design.dtype)))),
            1e-6,
            1.0 - 1e-6,
        )
        density = tf.maximum(
            tf.exp(-0.5 * tf.square(eta))
            / tf.sqrt(tf.constant(2.0 * 3.141592653589793, design.dtype)),
            1e-6,
        )
        weight = tf.square(density) / (probability * (1.0 - probability))
        working = tf.clip_by_value(
            eta + (response - probability) / density,
            -eta_clip,
            eta_clip,
        )
        precision = tf.einsum("bnk,bns,bnl->bskl", design, weight, design)
        precision += tf.eye(
            n_covariates,
            batch_shape=[tf.shape(design)[0], tf.shape(response)[2]],
            dtype=design.dtype,
        ) * float(prior_precision)
        right = tf.einsum("bnk,bns,bns->bsk", design, weight, working)
        solved = tf.linalg.solve(precision, right[..., None])[..., 0]
        beta = tf.transpose(
            tf.clip_by_value(solved, -coefficient_clip, coefficient_clip),
            [0, 2, 1],
        )
    if precision is None:
        raise RuntimeError("bounded probit IRLS did not execute")
    covariance = tf.linalg.inv(precision)
    scale = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariance), 1e-12))
    return tf.stop_gradient(beta), tf.stop_gradient(tf.transpose(scale, [0, 2, 1]))


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
