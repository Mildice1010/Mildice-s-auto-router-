"""Génération de grille, CI et table de connexions (partagé GUI / CLI)."""

from __future__ import annotations

import random
from typing import List

import numpy as np

from pathfinder import Connection

GRID_SIZE = 12
DEFAULT_COPPER_LAYERS = 2
MIN_COPPER_LAYERS = 1
MAX_COPPER_LAYERS = 32
N_COMPONENTS = 4
N_PINS = 4
MIN_CELL_GAP = 2
MIN_CONNECTIONS = 5
MAX_CONNECTIONS = 10


def pin_corner_offsets() -> np.ndarray:
    """P1…P4 aux quatre coins de la case du CI (coin inférieur gauche = origine)."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])


def ic_cells_too_close(a: tuple[int, int], b: tuple[int, int], min_gap: int) -> bool:
    ax, ay = a
    bx, by = b
    x_separated = (ax + 1 + min_gap <= bx) or (bx + 1 + min_gap <= ax)
    y_separated = (ay + 1 + min_gap <= by) or (by + 1 + min_gap <= ay)
    return not (x_separated or y_separated)


def random_ic_cells(
    n: int,
    grid_size: int,
    min_gap: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    margin = 2
    max_origin = grid_size - margin - 1
    candidates = [
        (x, y)
        for x in range(margin, max_origin)
        for y in range(margin, max_origin)
    ]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(not ic_cells_too_close(cell, c, min_gap) for c in chosen):
            chosen.append(cell)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(
            "Impossible de placer tous les composants — augmentez la grille ou réduisez l'écart."
        )
    return chosen


def random_connection_table(rng: random.Random) -> List[Connection]:
    n_links = rng.randint(MIN_CONNECTIONS, MAX_CONNECTIONS)
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    connections: List[Connection] = []

    while len(connections) < n_links:
        ic_a = rng.randrange(1, N_COMPONENTS + 1)
        ic_b = rng.randrange(1, N_COMPONENTS + 1)
        if ic_a == ic_b:
            continue
        pin_a = rng.randrange(1, N_PINS + 1)
        pin_b = rng.randrange(1, N_PINS + 1)
        key = tuple(sorted(((ic_a, pin_a), (ic_b, pin_b))))
        if key in seen:
            continue
        seen.add(key)
        connections.append(Connection(ic_a, pin_a, ic_b, pin_b))

    return connections
