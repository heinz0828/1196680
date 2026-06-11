"""Compatibility export for the active feature-hypergraph constructor.

The current MDHGNN implementation builds feature-level hyperedges inside
``models.mdhgnn.FeatureHypergraphConstructor``. It uses three active channels:
correlation soft hyperedges, domain-prior hyperedges, and adaptive hyperedges.
"""

from .mdhgnn import FeatureHypergraphConstructor


__all__ = ["FeatureHypergraphConstructor"]
