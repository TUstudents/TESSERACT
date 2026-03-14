"""Semantic backbone interfaces and rule-based baselines."""

from .datasets import NaturalLanguageTask, generate_nl_tasks
from .interface import Backbone, BackboneOutput
from .rule_based import RuleBasedBackbone

__all__ = [
    "Backbone",
    "BackboneOutput",
    "NaturalLanguageTask",
    "generate_nl_tasks",
    "RuleBasedBackbone",
]
