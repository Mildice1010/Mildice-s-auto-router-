"""Pathfinder multi-couches : broches dans A*, vias pondérés, rip-up des pistes."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

import numpy as np

Cell3D = Tuple[int, int, int]
Cell2D = Tuple[int, int]
Point2D = Tuple[float, float]

# Coûts de déplacement (entiers pour A*).
ORTHOGONAL_COST = 10
DIAGONAL_COST = 14
VIA_COST = 32
RIPUP_COST = 140


class Connection(NamedTuple):
    ic_a: int
    pin_a: int
    ic_b: int
    pin_b: int


@dataclass
class RouteFrame:
    """État d'une étape de visualisation."""

    phase: str  # explore | commit | rip | fail | done
    conn_index: int
    conn_total: int
    explored: Set[Cell3D] = field(default_factory=set)
    current: Optional[Cell3D] = None
    partial_path: List[Point2D] = field(default_factory=list)
    committed: List[Tuple[List[Point2D], int, int, List[Point2D]]] = field(
        default_factory=list
    )
    path_3d: List[Cell3D] = field(default_factory=list)
    layer: int = 0
    status: str = ""


def pin_xy(ic_cell: Tuple[int, int], pin: int, offsets: np.ndarray) -> Point2D:
    x, y = ic_cell
    dx, dy = offsets[pin - 1]
    return x + dx, y + dy


def pin_to_cell(pin: Point2D, grid_size: int) -> Cell2D:
    """Case de grille contenant la broche (coordonnées entières aux coins)."""
    xi = int(round(pin[0]))
    yi = int(round(pin[1]))
    xi = max(0, min(grid_size - 1, xi))
    yi = max(0, min(grid_size - 1, yi))
    return xi, yi


def all_pin_cells(
    ic_cells: List[Tuple[int, int]],
    offsets: np.ndarray,
    grid_size: int,
) -> Set[Cell2D]:
    """Toutes les cases de broches — toujours accessibles à l'A*."""
    cells: Set[Cell2D] = set()
    for ic in ic_cells:
        for pin in range(1, 5):
            cells.add(pin_to_cell(pin_xy(ic, pin, offsets), grid_size))
    return cells


def connection_pins(
    conn: Connection,
    ic_cells: List[Tuple[int, int]],
    offsets: np.ndarray,
) -> Tuple[Point2D, Point2D]:
    ic_a, ic_b = conn.ic_a - 1, conn.ic_b - 1
    return (
        pin_xy(ic_cells[ic_a], conn.pin_a, offsets),
        pin_xy(ic_cells[ic_b], conn.pin_b, offsets),
    )


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


def path_to_polyline(
    path_3d: List[Cell3D],
    start_pin: Point2D,
    end_pin: Optional[Point2D] = None,
) -> Tuple[List[Point2D], int]:
    """Trace affichée : broche → … → broche ; centres de case entre les deux."""
    if not path_3d:
        if end_pin is not None:
            return [start_pin, end_pin], 0
        return [start_pin], 0
    last = len(path_3d) - 1
    pts: List[Point2D] = []
    for i, (x, y, _) in enumerate(path_3d):
        if i == 0:
            pts.append(start_pin)
        elif end_pin is not None and i == last:
            pts.append(end_pin)
        else:
            pts.append((x + 0.5, y + 0.5))
    layers = [layer for _, _, layer in path_3d]
    dominant = max(set(layers), key=layers.count)
    return pts, dominant


def connection_octile_distance(
    conn: Connection,
    ic_cells: List[Tuple[int, int]],
    offsets: np.ndarray,
    grid_size: int,
) -> int:
    """Distance estimée entre broches (plus grand = connexion plus difficile)."""
    start_pin, end_pin = connection_pins(conn, ic_cells, offsets)
    start_xy = pin_to_cell(start_pin, grid_size)
    goal_xy = pin_to_cell(end_pin, grid_size)
    return octile_heuristic(start_xy, goal_xy)


def vias_along_path(path_3d: List[Cell3D]) -> List[Point2D]:
    """Positions 2D des vias (changements de couche)."""
    vias: List[Point2D] = []
    for i in range(1, len(path_3d)):
        x, y, layer = path_3d[i]
        if layer != path_3d[i - 1][2]:
            vias.append((x + 0.5, y + 0.5))
    return vias


class Pathfinder:
    YIELD_EVERY = 1

    def __init__(
        self,
        grid_size: int,
        copper_layers: int,
        ic_cells: List[Tuple[int, int]],
        offsets: np.ndarray,
        *,
        route_longest_first: bool = False,
        max_passes_multiplier: int = 4,
        adaptive_ripup: bool = False,
        corridor_ripup_on_fail: bool = False,
        global_reroute_on_stuck: bool = False,
        astar_reopen: bool = False,
        layer_heuristic_weight: int = 1,
        route_by_congestion: bool = True,
        prefer_assigned_layer: bool = False,
        preferred_layer_penalty: int = 24,
    ) -> None:
        self.grid_size = grid_size
        self.copper_layers = max(1, copper_layers)
        self.ic_cells = ic_cells
        self.offsets = offsets
        self.route_longest_first = route_longest_first
        self.max_passes_multiplier = max(1, max_passes_multiplier)
        self.adaptive_ripup = adaptive_ripup
        self.corridor_ripup_on_fail = corridor_ripup_on_fail
        self.global_reroute_on_stuck = global_reroute_on_stuck
        self.astar_reopen = astar_reopen
        self.layer_heuristic_weight = layer_heuristic_weight
        self.route_by_congestion = route_by_congestion
        self.prefer_assigned_layer = prefer_assigned_layer
        self.preferred_layer_penalty = preferred_layer_penalty
        self._ripup_cost = RIPUP_COST
        self.route_occupancy: List[Dict[Cell2D, int]] = [
            {} for _ in range(self.copper_layers)
        ]
        self.routes: Dict[int, List[Cell3D]] = {}

    def _corridor_congestion(
        self,
        idx: int,
        conn_pins: List[Tuple[Point2D, Point2D]],
    ) -> int:
        """Nombre de cases déjà occupées dans le couloir start→goal."""
        start_pin, end_pin = conn_pins[idx]
        sx, sy = pin_to_cell(start_pin, self.grid_size)
        gx, gy = pin_to_cell(end_pin, self.grid_size)
        x0, x1 = sorted((sx, gx))
        y0, y1 = sorted((sy, gy))
        score = 0
        for layer in range(self.copper_layers):
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    if self._owner((x, y), layer) is not None:
                        score += 1
        return score

    def _sort_pending(
        self,
        pending: List[int],
        connections: List[Connection],
        conn_pins: List[Tuple[Point2D, Point2D]],
    ) -> List[int]:
        if self.route_by_congestion:
            return sorted(
                pending,
                key=lambda i: self._corridor_congestion(i, conn_pins),
            )
        if self.route_longest_first:
            return sorted(
                pending,
                key=lambda i: connection_octile_distance(
                    connections[i], self.ic_cells, self.offsets, self.grid_size
                ),
                reverse=True,
            )
        return sorted(pending)

    def _owner(self, xy: Cell2D, layer: int) -> Optional[int]:
        return self.route_occupancy[layer].get(xy)

    def _is_walkable(self, cell: Cell3D) -> bool:
        """Toute case dans la grille est routable (broches jamais exclues)."""
        x, y, _layer = cell
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _ripup_penalty(self, xy: Cell2D, layer: int, conn_idx: int) -> int:
        owner = self._owner(xy, layer)
        if owner is None or owner == conn_idx:
            return 0
        return self._ripup_cost

    def _can_step(self, x: int, y: int, nx: int, ny: int, layer: int) -> bool:
        if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
            return False
        if not self._is_walkable((nx, ny, layer)):
            return False
        if nx != x and ny != y:
            if not self._is_walkable((x, ny, layer)):
                return False
            if not self._is_walkable((nx, y, layer)):
                return False
        return True

    def _move_cost(self, move: int, to_cell: Cell3D, conn_idx: int) -> int:
        x, y, layer = to_cell
        return move + self._ripup_penalty((x, y), layer, conn_idx)

    def _neighbors(
        self, cell: Cell3D, conn_idx: int
    ) -> Iterator[Tuple[Cell3D, int]]:
        x, y, layer = cell
        for dx, dy, cost in PLANE_STEPS:
            nx, ny = x + dx, y + dy
            if self._can_step(x, y, nx, ny, layer):
                nxt = (nx, ny, layer)
                yield nxt, self._move_cost(cost, nxt, conn_idx)
        for nl in (layer - 1, layer + 1):
            if 0 <= nl < self.copper_layers:
                via = (x, y, nl)
                if self._is_walkable(via):
                    yield via, self._move_cost(VIA_COST, via, conn_idx)

    def _remove_route(self, conn_idx: int) -> None:
        path = self.routes.pop(conn_idx, None)
        if not path:
            return
        for x, y, layer in path:
            if self.route_occupancy[layer].get((x, y)) == conn_idx:
                del self.route_occupancy[layer][(x, y)]

    def _rip_conflicts(self, path_3d: List[Cell3D], conn_idx: int) -> Set[int]:
        ripped: Set[int] = set()
        for x, y, layer in path_3d:
            owner = self._owner((x, y), layer)
            if owner is not None and owner != conn_idx:
                ripped.add(owner)
        for other in sorted(ripped, key=lambda o: len(self.routes.get(o, []))):
            self._remove_route(other)
        return ripped

    def _rip_blockers_in_corridor(
        self,
        start_xy: Cell2D,
        goal_xy: Cell2D,
        conn_idx: int,
        margin: int = 1,
    ) -> Set[int]:
        """Retire les pistes qui bloquent le couloir entre deux broches (après échec A*)."""
        x0, x1 = sorted((start_xy[0], goal_xy[0]))
        y0, y1 = sorted((start_xy[1], goal_xy[1]))
        ripped: Set[int] = set()
        for layer in range(self.copper_layers):
            for x in range(max(0, x0 - margin), min(self.grid_size, x1 + margin + 1)):
                for y in range(max(0, y0 - margin), min(self.grid_size, y1 + margin + 1)):
                    owner = self._owner((x, y), layer)
                    if owner is not None and owner != conn_idx:
                        ripped.add(owner)
        for other in sorted(ripped, key=lambda o: len(self.routes.get(o, []))):
            self._remove_route(other)
        return ripped

    def _occupy_path(self, path_3d: List[Cell3D], conn_idx: int) -> None:
        self.routes[conn_idx] = list(path_3d)
        for x, y, layer in path_3d:
            self.route_occupancy[layer][(x, y)] = conn_idx

    def _route_connection(
        self,
        idx: int,
        connections: List[Connection],
        conn_pins: List[Tuple[Point2D, Point2D]],
        total: int,
    ) -> Iterator[RouteFrame]:
        start_pin, end_pin = conn_pins[idx]
        start_xy = pin_to_cell(start_pin, self.grid_size)
        goal_xy = pin_to_cell(end_pin, self.grid_size)

        preferred_layer = idx % self.copper_layers if self.prefer_assigned_layer else None

        routed = False
        for frame in self.astar_iter(
            start_xy,
            goal_xy,
            start_pin,
            end_pin,
            idx,
            total,
            conn_pins,
            preferred_layer=preferred_layer,
        ):
            if frame.phase in ("commit", "rip"):
                routed = True
            yield frame

        if (
            not routed
            and self.corridor_ripup_on_fail
            and self._rip_blockers_in_corridor(start_xy, goal_xy, idx)
        ):
            for frame in self.astar_iter(
                start_xy,
                goal_xy,
                start_pin,
                end_pin,
                idx,
                total,
                conn_pins,
                preferred_layer=preferred_layer,
            ):
                if frame.phase in ("commit", "rip"):
                    routed = True
                yield frame

    def _committed_snapshot(
        self, conn_pins: List[Tuple[Point2D, Point2D]]
    ) -> List[Tuple[List[Point2D], int, int, List[Point2D]]]:
        out: List[Tuple[List[Point2D], int, int, List[Point2D]]] = []
        for idx in sorted(self.routes):
            path = self.routes[idx]
            sp, ep = conn_pins[idx]
            poly, layer = path_to_polyline(path, sp, ep)
            out.append((poly, layer, idx, vias_along_path(path)))
        return out

    def astar_iter(
        self,
        start_xy: Cell2D,
        goal_xy: Cell2D,
        start_pin: Point2D,
        end_pin: Point2D,
        conn_idx: int,
        conn_total: int,
        conn_pins: List[Tuple[Point2D, Point2D]],
        *,
        preferred_layer: Optional[int] = None,
    ) -> Iterator[RouteFrame]:
        explored: Set[Cell3D] = set()
        starts = [
            (start_xy[0], start_xy[1], layer) for layer in range(self.copper_layers)
        ]
        goals = {(goal_xy[0], goal_xy[1], layer) for layer in range(self.copper_layers)}

        open_heap: List[Tuple[int, int, Cell3D]] = []
        counter = 0
        g_score: dict[Cell3D, int] = {}
        came_from: dict[Cell3D, Cell3D] = {}

        for s in starts:
            if not self._is_walkable(s):
                continue
            g_score[s] = 0
            heapq.heappush(open_heap, (octile_heuristic(start_xy, goal_xy), counter, s))
            counter += 1

        if not open_heap:
            yield RouteFrame(
                phase="fail",
                conn_index=conn_idx,
                conn_total=conn_total,
                committed=self._committed_snapshot(conn_pins),
                status=(
                    f"Connexion {conn_idx + 1}/{conn_total} — broche de départ inaccessible"
                ),
            )
            return

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
                path_so_far = (
                    reconstruct(came_from, current) if current in came_from else [current]
                )
                partial, _ = path_to_polyline(path_so_far, start_pin)
                yield RouteFrame(
                    phase="explore",
                    conn_index=conn_idx,
                    conn_total=conn_total,
                    explored=set(explored),
                    current=current,
                    partial_path=partial,
                    committed=self._committed_snapshot(conn_pins),
                    layer=clayer,
                    status=f"Connexion {conn_idx + 1}/{conn_total} — exploration depuis broche",
                )

            if current in goals:
                path_3d = reconstruct(came_from, current)
                ripped = self._rip_conflicts(path_3d, conn_idx)
                self._occupy_path(path_3d, conn_idx)
                poly, dom_layer = path_to_polyline(path_3d, start_pin, end_pin)
                n_vias = len(vias_along_path(path_3d))
                status = (
                    f"Connexion {conn_idx + 1}/{conn_total} — piste posée "
                    f"(couche {dom_layer + 1}, {n_vias} via(s))"
                )
                if ripped:
                    nums = ", ".join(str(i + 1) for i in sorted(ripped))
                    status += f" — rip-up connexion(s) {nums}"
                yield RouteFrame(
                    phase="rip" if ripped else "commit",
                    conn_index=conn_idx,
                    conn_total=conn_total,
                    explored=set(explored),
                    current=current,
                    partial_path=poly,
                    committed=self._committed_snapshot(conn_pins),
                    path_3d=path_3d,
                    layer=dom_layer,
                    status=status,
                )
                return

            for neighbor, step_cost in self._neighbors(current, conn_idx):
                if neighbor in closed and not self.astar_reopen:
                    continue
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(neighbor, 10**9):
                    if neighbor in closed:
                        closed.discard(neighbor)
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    nx, ny, nl = neighbor
                    priority = (
                        tentative
                        + octile_heuristic((nx, ny), goal_xy)
                        + nl * self.layer_heuristic_weight
                    )
                    if (
                        preferred_layer is not None
                        and nl != preferred_layer
                    ):
                        priority += self.preferred_layer_penalty
                    heapq.heappush(open_heap, (priority, counter, neighbor))
                    counter += 1

        yield RouteFrame(
            phase="fail",
            conn_index=conn_idx,
            conn_total=conn_total,
            explored=explored,
            committed=self._committed_snapshot(conn_pins),
            status=f"Connexion {conn_idx + 1}/{conn_total} — échec (saturation / rip-up insuffisant)",
        )

    def route_all_iter(
        self,
        connections: List[Connection],
    ) -> Iterator[RouteFrame]:
        total = len(connections)
        conn_pins = [
            connection_pins(c, self.ic_cells, self.offsets) for c in connections
        ]

        max_passes = max(total * self.max_passes_multiplier, 1)
        passes = 0

        while passes < max_passes:
            pending = self._sort_pending(
                [i for i in range(total) if i not in self.routes],
                connections,
                conn_pins,
            )
            if not pending:
                break
            passes += 1
            if self.adaptive_ripup:
                self._ripup_cost = (
                    RIPUP_COST
                    if passes == 1
                    else max(RIPUP_COST // 2, VIA_COST + 1)
                )
            else:
                self._ripup_cost = RIPUP_COST

            for idx in pending:
                yield from self._route_connection(idx, connections, conn_pins, total)

        pending = [i for i in range(total) if i not in self.routes]
        if pending and self.global_reroute_on_stuck:
            for idx in list(self.routes):
                self._remove_route(idx)
            self._ripup_cost = max(RIPUP_COST // 2, VIA_COST + 1)
        failed = [i for i in range(total) if i not in self.routes]
        status = (
            f"Routage terminé — {len(self.routes)}/{total} connexion(s) "
            f"(via={VIA_COST}, rip-up={RIPUP_COST})"
        )
        if failed:
            nums = ", ".join(str(i + 1) for i in failed)
            status += f" — échec connexion(s) {nums}"
        yield RouteFrame(
            phase="done",
            conn_index=total,
            conn_total=total,
            committed=self._committed_snapshot(conn_pins),
            status=status,
        )


def route_connections_iter(
    connections: List[Connection],
    ic_cells: List[Tuple[int, int]],
    offsets: np.ndarray,
    grid_size: int,
    copper_layers: int,
    *,
    route_longest_first: bool = False,
    max_passes_multiplier: int = 4,
    adaptive_ripup: bool = False,
    corridor_ripup_on_fail: bool = False,
    global_reroute_on_stuck: bool = False,
    astar_reopen: bool = False,
    layer_heuristic_weight: int = 1,
    route_by_congestion: bool = True,
    prefer_assigned_layer: bool = False,
    preferred_layer_penalty: int = 24,
) -> Iterator[RouteFrame]:
    pf = Pathfinder(
        grid_size,
        copper_layers,
        ic_cells,
        offsets,
        route_longest_first=route_longest_first,
        max_passes_multiplier=max_passes_multiplier,
        adaptive_ripup=adaptive_ripup,
        corridor_ripup_on_fail=corridor_ripup_on_fail,
        global_reroute_on_stuck=global_reroute_on_stuck,
        astar_reopen=astar_reopen,
        layer_heuristic_weight=layer_heuristic_weight,
        route_by_congestion=route_by_congestion,
        prefer_assigned_layer=prefer_assigned_layer,
        preferred_layer_penalty=preferred_layer_penalty,
    )
    yield from pf.route_all_iter(connections)
