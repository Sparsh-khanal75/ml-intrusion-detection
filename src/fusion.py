"""
Entropy-weighted late fusion of classifier probabilities.

Idea (adapted from my IEEE IITCEE 2026 work on multi-model fusion for medical
image classification): when several models disagree, trust the one that is most
*certain* about this particular sample. Certainty is measured with normalised
Shannon entropy of each model's predicted class distribution:

    H_m(x) = -sum_k p_mk(x) log p_mk(x) / log K        (0 = certain, 1 = uniform)
    w_m(x) = (1 - H_m(x)) + eps,  normalised so sum_m w_m(x) = 1
    p(x)   = sum_m w_m(x) * p_m(x)

Unlike fixed-weight soft voting, the weights change per sample.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

EPS = 1e-6


def normalised_entropy(proba: np.ndarray) -> np.ndarray:
    k = proba.shape[1]
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(k)


def entropy_weighted_fusion(probas: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return fused probabilities (n, K) and the per-sample weights (n, M)."""
    stacked = np.stack(probas)                                   # (M, n, K)
    weights = np.stack([1.0 - normalised_entropy(p) + EPS for p in probas])  # (M, n)
    weights = weights / weights.sum(axis=0, keepdims=True)
    fused = (weights[:, :, None] * stacked).sum(axis=0)
    return fused, weights.T


class EntropyWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Fits each base pipeline, then fuses their predict_proba outputs per sample."""

    def __init__(self, estimators: Dict[str, BaseEstimator]):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = {name: clone(est).fit(X, y) for name, est in self.estimators.items()}
        self.classes_ = next(iter(self.fitted_.values())).classes_
        return self

    @classmethod
    def from_fitted(cls, fitted: Dict[str, BaseEstimator]) -> "EntropyWeightedEnsemble":
        """Wrap already-trained models without retraining them."""
        ens = cls(estimators=fitted)
        ens.fitted_ = dict(fitted)
        ens.classes_ = next(iter(fitted.values())).classes_
        return ens

    def member_probas(self, X) -> Dict[str, np.ndarray]:
        return {name: est.predict_proba(X) for name, est in self.fitted_.items()}

    def predict_proba(self, X):
        fused, self.last_weights_ = entropy_weighted_fusion(list(self.member_probas(X).values()))
        return fused

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
