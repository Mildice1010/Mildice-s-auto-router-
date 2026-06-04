"""Exécution headless d'une simulation de routage."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from layout import (
    DEFAULT_COPPER_LAYERS,
    GRID_SIZE,
    MIN_CELL_GAP,
    N_COMPONENTS,
    pin_corner_offsets,
    random_connection_table,
    random_ic_cells,
)
from pathfinder import Connection, Pathfinder


@dataclass(frozen=True)
class RoutingResult:
    seed: Optional[int]
    total_connections: int
    succeeded: int
    failed: int
    rip_events: int

    @property
    def success_rate(self) -> float:
        if self.total_connections == 0:
            return 0.0
        return self.succeeded / self.total_connections


def run_routing(
    *,
    seed: Optional[int] = None,
    grid_size: int = GRID_SIZE,
    copper_layers: int = DEFAULT_COPPER_LAYERS,
    n_components: int = N_COMPONENTS,
    min_cell_gap: int = MIN_CELL_GAP,
    connections: Optional[List[Connection]] = None,
    ic_cells: Optional[List[tuple[int, int]]] = None,
) -> RoutingResult:
    rng = random.Random(seed)
    offsets = pin_corner_offsets()
    cells = ic_cells or random_ic_cells(n_components, grid_size, min_cell_gap, rng)
    conns = connections or random_connection_table(rng)

    pf = Pathfinder(grid_size, copper_layers, cells, offsets)
    rip_events = 0
    for frame in pf.route_all_iter(conns):
        if frame.phase == "rip":
            rip_events += 1

    total = len(conns)
    succeeded = len(pf.routes)
    return RoutingResult(
        seed=seed,
        total_connections=total,
        succeeded=succeeded,
        failed=total - succeeded,
        rip_events=rip_events,
    )


def run_routing_task(params: dict) -> RoutingResult:
    """Point d'entrée pour exécution parallèle (arguments sérialisables)."""
    return run_routing(**params)
