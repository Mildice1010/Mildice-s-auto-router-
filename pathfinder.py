"""Pathfinder multi-couches pour router les connexions sur la grille."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple

import numpy as np

Cell3D = Tuple[int, int, int]
Cell2D = Tuple[int, int]


class Connection(NamedTuple):
    ic_a: int
    pin_a: int
    ic_b: int
    pin_b: int


@dataclass
class RouteFrame:
    """État d'une étape de visualisation."""

    phase: str  # explore | commit | fail | done
    conn_index: int
    conn_total: int
    explored: Set[Cell3D] = field(default_factory=set)
    current: Optional[Cell3D] = None
    partial_path: List[Cell2D] = field(default_factory=list)
    committed: List[Tuple[List[Cell2D], int, int]] = field(default_factory=list)
    layer: int = 0
    status: str = ""


def pin_xy(center: Tuple[int, int], pin: int, offsets: np.ndarray) -> Tuple[float, float]:
    x, y = center
    dx, dy = offsets[pin - 1]
    return x + dx, y + dy


def snap_to_grid(x: float, y: float, grid_size: int) -> Cell2D:
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(grid_size - 1, xi))
    yi = max(0, min(grid_size - 1, yi))
    return xi, yi


def component_obstacles(centers: List[Tuple[int, int]]) -> Set[Cell2D]:
    return {(cx, cy) for cx, cy in centers}


def connection_endpoints(
    conn: Connection,
    centers: List[Tuple[int, int]],
    offsets: np.ndarray,
    grid_size: int,
) -> Tuple[Cell2D, Cell2D]:
    ic_a, ic_b = conn.ic_a - 1, conn.ic_b - 1
    ax, ay = pin_xy(centers[ic_a], conn.pin_a, offsets)
    bx, by = pin_xy(centers[ic_b], conn.pin_b, offsets)
    return snap_to_grid(ax, ay, grid_size), snap_to_grid(bx, by, grid_size)


# Coûts entiers (orthogonal=10, diagonal≈14) pour A* avec diagonales.
ORTHOGONAL_COST = 10
DIAGONAL_COST = 14

# (dx, dy, coût) — 8 directions horizontales/verticales/diagonales
PLANE_STEPS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, ORTHOGONAL_COST),
    (-1, 0, ORTHOGONAL_COST),
    (0, 1, ORTHOGONAL_COST),
    (0, -1, ORTHOGONAL_COST),
    (1, 1, DIAGONAL_COST),
    (1, -1, DIAGONAL_COST),
    (-1, 1, DIAGONAL_COST),
    (-1, -1, DIAGONAL_COST),
)


def octile_heuristic(a: Cell2D, b: Cell2D) -> int:
    """Heuristique admissible pour déplacements en 8 directions."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return ORTHOGONAL_COST * (dx + dy) + (DIAGONAL_COST - 2 * ORTHOGONAL_COST) * min(dx, dy)


def reconstruct(came_from: dict, current: Cell3D) -> List[Cell3D]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_to_polyline(path_3d: List[Cell3D]) -> Tuple[List[Cell2D], int]:
    """Trace 2D pour affichage (centres de cellules) + couche dominante."""
    if not path_3d:
        return [], 0
    poly = [(x + 0.5, y + 0.5) for x, y, _ in path_3d]
    layers = [layer for _, _, layer in path_3d]
    dominant = max(set(layers), key=layers.count)
    return poly, dominant


class Pathfinder:
    VIA_COST = 2
    YIELD_EVERY = 1

    def __init__(
        self,
        grid_size: int,
        copper_layers: int,
        centers: List[Tuple[int, int]],
        offsets: np.ndarray,
    ) -> None:
        self.grid_size = grid_size
        self.copper_layers = max(1, copper_layers)
        self.centers = centers
        self.offsets = offsets
        self.blocked = component_obstacles(centers)
        self.occupancy: List[Set[Cell2D]] = [set() for _ in range(self.copper_layers)]

    def _cell_free(self, xy: Cell2D, layer: int, endpoints: Set[Cell2D]) -> bool:
        if xy in self.blocked and xy not in endpoints:
            return False
        if xy in self.occupancy[layer] and xy not in endpoints:
            return False
        return True

    def _is_walkable(self, cell: Cell3D, endpoints: Set[Cell2D]) -> bool:
        x, y, layer = cell
        return self._cell_free((x, y), layer, endpoints)

    def _can_step(
        self,
        x: int,
        y: int,
        nx: int,
        ny: int,
        layer: int,
        endpoints: Set[Cell2D],
    ) -> bool:
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return False
        if not self._cell_free((nx, ny), layer, endpoints):
            return False
        # Pas de coupe de coin à travers un obstacle orthogonal.
        if nx != x and ny != y:
            if not self._cell_free((x, ny), layer, endpoints):
                return False
            if not self._cell_free((nx, y), layer, endpoints):
                return False
        return True

    def _neighbors(self, cell: Cell3D, endpoints: Set[Cell2D]) -> Iterator[Tuple[Cell3D, int]]:
        x, y, layer = cell
        for dx, dy, cost in PLANE_STEPS:
            nx, ny = x + dx, y + dy
            if self._can_step(x, y, nx, ny, layer, endpoints):
                yield (nx, ny, layer), cost
        for nl in (layer - 1, layer + 1):
            if 0 <= nl < self.copper_layers:
                via = (x, y, nl)
                if self._is_walkable(via, endpoints):
                    yield via, self.VIA_COST

    def astar_iter(
        self,
        start_xy: Cell2D,
        goal_xy: Cell2D,
        endpoints: Set[Cell2D],
        conn_index: int,
        conn_total: int,
        committed: List[Tuple[List[Cell2D], int, int]],
    ) -> Iterator[RouteFrame]:
        explored: Set[Cell3D] = set()
        starts = [(start_xy[0], start_xy[1], layer) for layer in range(self.copper_layers)]
        goals = {(goal_xy[0], goal_xy[1], layer) for layer in range(self.copper_layers)}

        open_heap: List[Tuple[int, int, Cell3D]] = []
        counter = 0
        g_score: dict[Cell3D, int] = {}
        came_from: dict[Cell3D, Cell3D] = {}

        for s in starts:
            if not self._is_walkable(s, endpoints):
                continue
            g_score[s] = 0
            priority = octile_heuristic(start_xy, goal_xy)
            heapq.heappush(open_heap, (priority, counter, s))
            counter += 1

        closed: Set[Cell3D] = set()
        expansions = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            explored.add(current)
            expansions += 1
            cx, cy, clayer = current

            if expansions % self.YIELD_EVERY == 0:
                partial, _ = path_to_polyline(reconstruct(came_from, current) if current in came_from else [current])
                yield RouteFrame(
                    phase="explore",
                    conn_index=conn_index,
                    conn_total=conn_total,
                    explored=set(explored),
                    current=current,
                    partial_path=partial,
                    committed=list(committed),
                    layer=clayer,
                    status=f"Connexion {conn_index + 1}/{conn_total} — exploration A*",
                )

            if current in goals:
                path_3d = reconstruct(came_from, current)
                poly, dom_layer = path_to_polyline(path_3d)
                for x, y, layer in path_3d:
                    if (x, y) not in endpoints:
                        self.occupancy[layer].add((x, y))
                yield RouteFrame(
                    phase="commit",
                    conn_index=conn_index,
                    conn_total=conn_total,
                    explored=set(explored),
                    current=current,
                    partial_path=poly,
                    committed=list(committed),
                    layer=dom_layer,
                    status=f"Connexion {conn_index + 1}/{conn_total} — piste posée (couche {dom_layer + 1})",
                )
                return

            for neighbor, move_cost in self._neighbors(current, endpoints):
                if neighbor in closed:
                    continue
                tentative = g_score[current] + move_cost
                if tentative < g_score.get(neighbor, 10**9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    nx, ny, nl = neighbor
                    priority = tentative + octile_heuristic((nx, ny), goal_xy) + int(nl * 0.1)
                    heapq.heappush(open_heap, (priority, counter, neighbor))
                    counter += 1

        yield RouteFrame(
            phase="fail",
            conn_index=conn_index,
            conn_total=conn_total,
            explored=explored,
            committed=list(committed),
            status=f"Connexion {conn_index + 1}/{conn_total} — échec (grille saturée)",
        )

    def route_all_iter(
        self,
        connections: List[Connection],
    ) -> Iterator[RouteFrame]:
        committed: List[Tuple[List[Cell2D], int, int]] = []
        total = len(connections)

        for idx, conn in enumerate(connections):
            start_xy, goal_xy = connection_endpoints(
                conn, self.centers, self.offsets, self.grid_size
            )
            endpoints = {start_xy, goal_xy}

            routed = False
            for frame in self.astar_iter(
                start_xy, goal_xy, endpoints, idx, total, committed
            ):
                if frame.phase == "commit":
                    committed.append((frame.partial_path, frame.layer, idx))
                    frame.committed = list(committed)
                    routed = True
                yield frame

            if not routed:
                continue

        yield RouteFrame(
            phase="done",
            conn_index=total,
            conn_total=total,
            committed=list(committed),
            status=f"Routage terminé — {len(committed)}/{total} connexion(s)",
        )


def route_connections_iter(
    connections: List[Connection],
    centers: List[Tuple[int, int]],
    offsets: np.ndarray,
    grid_size: int,
    copper_layers: int,
) -> Iterator[RouteFrame]:
    pf = Pathfinder(grid_size, copper_layers, centers, offsets)
    yield from pf.route_all_iter(connections)
