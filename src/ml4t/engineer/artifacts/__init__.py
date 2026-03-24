"""Artifact contracts owned by ml4t-engineer."""

from .features import FeatureDefinition, FeatureSchema, FeatureSpec
from .labels import LabelDefinition, LabelSchema, LabelSpec
from .predictions import PredictionDefinition, PredictionSchema, PredictionSpec

__all__ = [
    "FeatureDefinition",
    "FeatureSchema",
    "FeatureSpec",
    "LabelDefinition",
    "LabelSchema",
    "LabelSpec",
    "PredictionDefinition",
    "PredictionSchema",
    "PredictionSpec",
]
