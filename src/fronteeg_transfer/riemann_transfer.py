"""Riemannian EEG transfer adapted from the MIT-licensed mindscape project.

The zero-shot method estimates OAS covariance matrices, re-centres every source
subject and target subject independently on the SPD manifold, maps the aligned
matrices to a tangent space, and fits logistic regression on source labels only.
Target labels are never consumed by the method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class CovarianceDomain:
    covariances: np.ndarray
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None


def estimate_covariances(eeg: np.ndarray, estimator: str = "oas") -> np.ndarray:
    """Estimate one robust spatial covariance matrix per EEG window."""
    values = np.asarray(eeg, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("eeg must have shape [windows, channels, samples]")
    return np.asarray(Covariances(estimator=estimator).transform(values))


def recenter_covariances(covariances: np.ndarray) -> np.ndarray:
    """Transport one domain's covariance mean to the identity matrix."""
    values = np.asarray(covariances, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariances must have shape [windows, channels, channels]")
    whitener = invsqrtm(mean_riemann(values))
    return np.einsum("ij,njk,kl->nil", whitener, values, whitener)


def align_to_reference(
    covariances: np.ndarray,
    reference_covariances: np.ndarray,
    *,
    mean: str,
) -> np.ndarray:
    """Whiten covariances using a separately supplied, unlabeled reference domain."""
    values = np.asarray(covariances, dtype=np.float64)
    reference = np.asarray(reference_covariances, dtype=np.float64)
    if values.ndim != 3 or reference.ndim != 3:
        raise ValueError("covariances and reference must both be rank-three")
    if values.shape[1:] != reference.shape[1:]:
        raise ValueError("covariances and reference must have matching channel dimensions")
    if mean == "euclidean":
        center = reference.mean(axis=0)
    elif mean == "riemann":
        center = mean_riemann(reference)
    else:
        raise ValueError(f"unknown covariance mean {mean}")
    whitener = invsqrtm(center)
    return np.einsum("ij,njk,kl->nil", whitener, values, whitener)


def align_by_group(
    covariances: np.ndarray,
    groups: np.ndarray,
    *,
    mean: str,
) -> np.ndarray:
    """Align each source subject independently with Euclidean or Riemannian means."""
    values = np.asarray(covariances, dtype=np.float64)
    domain_ids = np.asarray(groups)
    if len(values) != len(domain_ids):
        raise ValueError("covariances and groups must contain the same number of windows")
    output = np.empty_like(values)
    for domain_id in np.unique(domain_ids):
        mask = domain_ids == domain_id
        output[mask] = align_to_reference(values[mask], values[mask], mean=mean)
    return output


def recenter_by_group(covariances: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Re-centre each subject independently without using labels."""
    values = np.asarray(covariances, dtype=np.float64)
    domain_ids = np.asarray(groups)
    if len(values) != len(domain_ids):
        raise ValueError("covariances and groups must contain the same number of windows")
    output = np.empty_like(values)
    for domain_id in np.unique(domain_ids):
        mask = domain_ids == domain_id
        output[mask] = recenter_covariances(values[mask])
    return output


def make_tangent_classifier(*, class_weight: str | None = "balanced"):
    """Build the tangent-space plus logistic-regression readout."""
    return make_pipeline(
        TangentSpace(metric="riemann"),
        LogisticRegression(
            max_iter=2_000,
            C=1.0,
            class_weight=class_weight,
            random_state=2026,
        ),
    )


def fit_source_classifier(source: CovarianceDomain, *, recenter: bool):
    """Fit a source-only classifier, optionally aligning every source group."""
    if source.labels is None:
        raise ValueError("source labels are required")
    covariances = source.covariances
    if recenter:
        if source.groups is None:
            raise ValueError("source groups are required for re-centering")
        covariances = recenter_by_group(covariances, source.groups)
    return make_tangent_classifier().fit(covariances, source.labels)


def predict_target(classifier, target: CovarianceDomain, *, recenter: bool) -> np.ndarray:
    """Predict target probabilities; re-centering consumes signals but no labels."""
    covariances = target.covariances
    if recenter:
        covariances = (
            recenter_covariances(covariances)
            if target.groups is None
            else recenter_by_group(covariances, target.groups)
        )
    return np.asarray(classifier.predict_proba(covariances), dtype=np.float64)
