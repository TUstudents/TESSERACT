"""Semantic backbone interfaces and baseline implementations."""

from .datasets import NaturalLanguageTask, generate_nl_tasks
from .interface import Backbone, BackboneOutput
from .learned import (
    BackboneTrainingBatch,
    BackboneVocabulary,
    CanonicalPromptVocabulary,
    LearnedBackbone,
    LearnedBackboneModel,
    build_backbone_training_batch,
    build_learned_backbone,
    train_backbone_step,
)
from .rule_based import RuleBasedBackbone

__all__ = [
    "Backbone",
    "BackboneOutput",
    "NaturalLanguageTask",
    "generate_nl_tasks",
    "BackboneTrainingBatch",
    "BackboneVocabulary",
    "CanonicalPromptVocabulary",
    "LearnedBackbone",
    "LearnedBackboneModel",
    "build_backbone_training_batch",
    "build_learned_backbone",
    "train_backbone_step",
    "RuleBasedBackbone",
]
