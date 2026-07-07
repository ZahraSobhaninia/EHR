"""
Helper functions for PartitionSHAP explainability.

Place this file at:
    corebehrt/modules/explainability/partition_tree.py

Usage (from trainer.py):
    from corebehrt.modules.explainability.partition_tree import (
        get_token_groups,
        build_partition_tree,
    )
"""

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def get_token_groups(token_ids, id_to_code, special_group_name="Special"):
    """
    Maps each token in a sequence to its MDPS group based on the
    vocabulary code prefix (M/, D/, P/, S/). Anything without a
    recognized prefix (CLS, SEP, PAD, DOB, etc.) falls into "Special".

    Args:
        token_ids: list or 1D array of token ids (length = seq_len)
        id_to_code: dict mapping token_id -> vocab code string
        special_group_name: label used for non-MDPS tokens

    Returns:
        List[str] of group labels, same length as token_ids
    """
    groups = []
    for tid in token_ids:
        code = id_to_code.get(tid, None)
        if code is None:
            groups.append(special_group_name)
            continue

        if code.startswith("M/"):
            groups.append("Medication")
        elif code.startswith("D/"):
            groups.append("Diagnosis")
        elif code.startswith("P/"):
            groups.append("Procedure")
        elif code.startswith("S/"):
            groups.append("Special_Surgery")
        else:
            groups.append(special_group_name)

    return groups


def build_partition_tree(groups, method="complete"):
    """
    Builds a linkage matrix (scipy hierarchical clustering format)
    from a list of group labels, suitable for shap.maskers.Partition.

    Tokens in the same group get distance 0, tokens in different
    groups get distance 1. This is O(n^2) — fine for seq_len up to
    a few thousand, but consider shap.utils.partition_tree for very
    long sequences if this becomes a bottleneck.

    Args:
        groups: list of group labels, one per token position
        method: linkage method passed to scipy (default "complete")

    Returns:
        clustering: linkage matrix to pass into shap.maskers.Partition
    """
    n = len(groups)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = 0.0 if groups[i] == groups[j] else 1.0

    condensed = squareform(distance_matrix, checks=False)
    clustering = linkage(condensed, method=method)
    return clustering