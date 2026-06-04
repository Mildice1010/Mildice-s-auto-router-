#!/usr/bin/env python3
"""Compare baseline vs pathfinder amélioré sur les mêmes plateaux."""

from __future__ import annotations

import random
from dataclasses import dataclass

from layout import (
    DEFAULT_COPPER_LAYERS,
    GRID_SIZE,
    pin_corner_offsets,
    random_connection_table,
    random_ic_cells,
)
from pathfinder import Pathfinder


@dataclass
class BenchStats:
    label: str
    n_boards: int
    total_connections: int
    succeeded: int
    failed: int
    rip_events: int
    mean_per_board_rate: float = 0.0

    @property
    def success_rate_pct(self) -> float:
        if self.total_connections == 0:
            return 0.0
        return 100.0 * self.succeeded / self.total_connections

    @property
    def mean_per_board_pct(self) -> float:
        return 100.0 * self.mean_per_board_rate


def _run_batch(
    label: str,
    n_boards: int,
    base_seed: int,
    *,
    max_passes_multiplier: int,
    adaptive_ripup: bool,
    corridor_ripup_on_fail: bool,
    global_reroute_on_stuck: bool,
    astar_reopen: bool,
    layer_heuristic_weight: int,
    route_by_congestion: bool,
) -> BenchStats:
    offsets = pin_corner_offsets()
    total_conns = 0
    succeeded = 0
    rip_events = 0
    board_rates: list[float] = []

    for i in range(n_boards):
        seed = base_seed + i
        rng = random.Random(seed)
        cells = random_ic_cells(4, GRID_SIZE, 2, rng)
        conns = random_connection_table(rng)
        pf = Pathfinder(
            GRID_SIZE,
            DEFAULT_COPPER_LAYERS,
            cells,
            offsets,
            max_passes_multiplier=max_passes_multiplier,
            adaptive_ripup=adaptive_ripup,
            corridor_ripup_on_fail=corridor_ripup_on_fail,
            global_reroute_on_stuck=global_reroute_on_stuck,
            astar_reopen=astar_reopen,
            layer_heuristic_weight=layer_heuristic_weight,
            route_by_congestion=route_by_congestion,
        )
        rips = 0
        for frame in pf.route_all_iter(conns):
            if frame.phase == "rip":
                rips += 1
        ok = len(pf.routes)
        n = len(conns)
        total_conns += n
        succeeded += ok
        rip_events += rips
        board_rates.append(ok / n if n else 0.0)

    return BenchStats(
        label=label,
        n_boards=n_boards,
        total_connections=total_conns,
        succeeded=succeeded,
        failed=total_conns - succeeded,
        rip_events=rip_events,
        mean_per_board_rate=sum(board_rates) / len(board_rates) if board_rates else 0.0,
    )


def main() -> None:
    n_boards = 200
    base_seed = 0

    legacy = _run_batch(
        "baseline (multi-couches, vias libres)",
        n_boards,
        base_seed,
        max_passes_multiplier=4,
        adaptive_ripup=False,
        corridor_ripup_on_fail=False,
        global_reroute_on_stuck=False,
        astar_reopen=False,
        layer_heuristic_weight=1,
        route_by_congestion=False,
    )
    improved = _run_batch(
        "amélioré (ordre par congestion + rip court d'abord)",
        n_boards,
        base_seed,
        max_passes_multiplier=4,
        adaptive_ripup=False,
        corridor_ripup_on_fail=False,
        global_reroute_on_stuck=False,
        astar_reopen=False,
        layer_heuristic_weight=1,
        route_by_congestion=True,
    )

    delta = improved.success_rate_pct - legacy.success_rate_pct

    print(f"Benchmark — {n_boards} plateaux (seeds {base_seed}…{base_seed + n_boards - 1})")
    print()
    for s in (legacy, improved):
        print(f"  {s.label}")
        print(f"    taux global      : {s.success_rate_pct:.2f} %")
        print(f"    moyenne / plateau: {s.mean_per_board_pct:.2f} %")
        print(f"    ratées           : {s.failed} / {s.total_connections}")
        print(f"    rip-up           : {s.rip_events}")
        print()

    print(f"  Δ taux global : {delta:+.2f} pt")
    if delta > 0:
        print("\n  → L'amélioration est meilleure.")
    elif delta < 0:
        print("\n  → L'amélioration est moins bonne.")
    else:
        print("\n  → Égalité.")


if __name__ == "__main__":
    main()
