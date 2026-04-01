# CV/frameworks/CTR-GCN/graph/coco17.py
from __future__ import annotations

import numpy as np

# COCO-17 keypoint indices (common order):
# 0 nose
# 1 left_eye, 2 right_eye
# 3 left_ear, 4 right_ear
# 5 left_shoulder, 6 right_shoulder
# 7 left_elbow, 8 right_elbow
# 9 left_wrist, 10 right_wrist
# 11 left_hip, 12 right_hip
# 13 left_knee, 14 right_knee
# 15 left_ankle, 16 right_ankle


def _edge_to_adjacency(num_node: int, edge: list[tuple[int, int]], self_link: bool = True) -> np.ndarray:
    A = np.zeros((num_node, num_node), dtype=np.float32)

    if self_link:
        for i in range(num_node):
            A[i, i] = 1.0

    for i, j in edge:
        A[j, i] = 1.0  # j <- i (consistent with many ST-GCN/CTR-GCN graph defs)
        A[i, j] = 1.0  # undirected

    return A


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    # Column-normalize (common in ST-GCN style)
    Dl = np.sum(A, axis=0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn


class Graph:
    """
    Minimal Graph implementation compatible with common CTR-GCN forks.
    Exposes:
      - self.A : adjacency matrix (K, V, V) or (V, V)

    Many CTR-GCN repos accept a single adjacency (V,V) OR a stack (K,V,V).
    We'll provide K=1 to keep it simple and compatible.
    """
    def __init__(self, labeling_mode: str = "spatial", **kwargs):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]

        # COCO skeleton edges (a standard, simple tree-ish structure)
        # head
        neighbor = [
            (0, 1), (0, 2),
            (1, 3), (2, 4),

            # shoulders / torso
            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),

            # legs
            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]
        # Build 3-subset adjacency like ST-GCN / many CTR-GCN forks:
        #   A[0] = self-links
        #   A[1] = inward edges
        #   A[2] = outward edges

        self_links = [(i, i) for i in range(self.num_node)]

        # For inward/outward we need a directed definition.
        # We'll treat edges as parent->child in a simple COCO tree-ish skeleton.
        # inward:  j <- i (child receives from parent)
        inward = [
            (0, 1), (0, 2),
            (1, 3), (2, 4),

            (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),

            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]
        outward = [(j, i) for (i, j) in inward]

        A0 = _edge_to_adjacency(self.num_node, self_links, self_link=False)   # self only
        A1 = _edge_to_adjacency(self.num_node, inward, self_link=False)       # inward only
        A2 = _edge_to_adjacency(self.num_node, outward, self_link=False)      # outward only

        A0 = _normalize_digraph(A0)
        A1 = _normalize_digraph(A1)
        A2 = _normalize_digraph(A2)

        self.A = np.stack([A0, A1, A2], axis=0).astype(np.float32)  # (3, V, V)
