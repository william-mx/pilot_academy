from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import tensorflow as tf


LossLike = Union[str, tf.keras.losses.Loss, Callable[..., tf.Tensor]]
MetricLike = Union[str, tf.keras.metrics.Metric]
OptimizerLike = tf.keras.optimizers.Optimizer


@dataclass
class TrainingSpec:
    """
    TensorFlow/Keras training specification.

    This is a pure data container for everything needed to:
      - model.compile(...)
      - model.fit(...)

    Create instances via a factory (e.g. from Hydra cfg) and keep this class
    free of config parsing and side effects.
    """

    # --- compile() ---
    optimizer: OptimizerLike
    loss: LossLike
    metrics: List[MetricLike] = field(default_factory=list)

    # Extra compile kwargs (e.g. run_eagerly, jit_compile, weighted_metrics, etc.)
    compile_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- fit() ---
    epochs: int = 1
    steps_per_epoch: Optional[int] = None
    validation_steps: Optional[int] = None

    callbacks: List[tf.keras.callbacks.Callback] = field(default_factory=list)

    # Extra fit kwargs (e.g. verbose, initial_epoch, class_weight, max_queue_size, etc.)
    fit_kwargs: Dict[str, Any] = field(default_factory=dict)

    def compile_model(self, model: tf.keras.Model) -> None:
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
            **self.compile_kwargs,
        )

    def fit(
        self,
        model: tf.keras.Model,
        train_ds,
        val_ds=None,
    ):
        return model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks=self.callbacks,
            **self.fit_kwargs,
        )
