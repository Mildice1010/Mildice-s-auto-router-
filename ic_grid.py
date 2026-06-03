#!/usr/bin/env python3
"""Affiche 4 CI à 4 broches sur une grille et route les connexions par pathfinding."""

import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from pathfinder import Connection, RouteFrame, route_connections_iter

GRID_SIZE = 12
DEFAULT_COPPER_LAYERS = 2
MIN_COPPER_LAYERS = 1
MAX_COPPER_LAYERS = 32
N_COMPONENTS = 4
N_PINS = 4
PIN_HALF = 0.18
MIN_CELL_GAP = 2
MIN_CONNECTIONS = 5
MAX_CONNECTIONS = 10
ROUTE_TIMER_MS = 35

LAYER_TRACE_COLORS = [
    "#d35400",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#c0392b",
    "#16a085",
    "#f39c12",
    "#2c3e50",
]
CONN_COLORS = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#1abc9c"]


def pin_offsets(half: float = PIN_HALF) -> np.ndarray:
    """Quatre broches aux coins d'un petit carré (P1…P4)."""
    return np.array([
        [-half, -half],
        [half, -half],
        [half, half],
        [-half, half],
    ])


def random_centers(
    n: int,
    grid_size: int,
    min_gap: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Positions entières sur la grille, sans centres trop proches."""
    margin = 2
    candidates = [
        (x, y)
        for x in range(margin, grid_size - margin)
        for y in range(margin, grid_size - margin)
    ]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - c[0]) >= min_gap and abs(cell[1] - c[1]) >= min_gap for c in chosen):
            chosen.append(cell)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(
            "Impossible de placer tous les composants — augmentez GRID_SIZE ou réduisez MIN_CELL_GAP."
        )
    return chosen


def random_connection_table(rng: random.Random) -> List[Connection]:
    """Génère des paires de broches à relier entre CI distincts."""
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


def parse_copper_layers(text: str) -> Optional[int]:
    try:
        n = int(text.strip())
    except ValueError:
        return None
    if MIN_COPPER_LAYERS <= n <= MAX_COPPER_LAYERS:
        return n
    return None


def format_connection_table(
    connections: List[Connection],
    copper_layers: int,
    route_status: str = "",
) -> str:
    header = (
        f"Couches de cuivre : {copper_layers}\n"
        "Table de connexions\n"
        + "─" * 22
    )
    rows = [f"CI{c.ic_a}-P{c.pin_a}  ↔  CI{c.ic_b}-P{c.pin_b}" for c in connections]
    text = header + "\n" + "\n".join(rows)
    if route_status:
        text += "\n\n" + route_status
    return text


def pin_xy(center: tuple[int, int], pin: int, offsets: np.ndarray) -> tuple[float, float]:
    x, y = center
    dx, dy = offsets[pin - 1]
    return x + dx, y + dy


def main() -> None:
    rng = random.Random()
    colors = CONN_COLORS[:N_COMPONENTS]
    centers = random_centers(N_COMPONENTS, GRID_SIZE, MIN_CELL_GAP, rng)
    offsets = pin_offsets()

    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.16, wspace=0.28)

    ax_grid = fig.add_axes((0.06, 0.18, 0.58, 0.74))
    ax_table = fig.add_axes((0.68, 0.18, 0.28, 0.74))
    ax_copper = fig.add_axes((0.68, 0.125, 0.28, 0.04))
    ax_btn_regen = fig.add_axes((0.68, 0.03, 0.28, 0.038))
    ax_btn_route = fig.add_axes((0.68, 0.075, 0.28, 0.038))

    fig.text(0.68, 0.17, "Couches de cuivre", fontsize=9, va="bottom", ha="left")

    ax_grid.set_xlim(0, GRID_SIZE)
    ax_grid.set_ylim(0, GRID_SIZE)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks(np.arange(0, GRID_SIZE + 1))
    ax_grid.set_yticks(np.arange(0, GRID_SIZE + 1))
    ax_grid.grid(True, which="both", linestyle="-", linewidth=0.6, color="#888888", alpha=0.7)
    ax_grid.set_facecolor("#f8f8f8")

    for i, (cx, cy) in enumerate(centers):
        pins = offsets + np.array([cx, cy])
        ax_grid.scatter(
            pins[:, 0],
            pins[:, 1],
            s=120,
            c=colors[i],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"CI {i + 1}",
        )
        ax_grid.scatter(cx, cy, s=30, c=colors[i], alpha=0.35, zorder=4)
        for pin_idx in range(N_PINS):
            px, py = pin_xy((cx, cy), pin_idx + 1, offsets)
            ax_grid.annotate(
                f"P{pin_idx + 1}",
                (px, py),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=colors[i],
            )

    ax_grid.legend(loc="upper left", framealpha=0.9, fontsize=8)

    ax_table.axis("off")
    ax_table.set_facecolor("#fafafa")
    table_text = ax_table.text(
        0.05,
        0.95,
        "",
        transform=ax_table.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        linespacing=1.45,
    )

    wire_artists: list = []
    route_artists: list = []
    route_timer = None
    route_gen = None

    ui_state = {
        "connections": [],
        "copper_layers": DEFAULT_COPPER_LAYERS,
        "route_status": "",
        "routing": False,
    }

    def set_title(status: str = "") -> None:
        base = "Grille PCB — composants et routage"
        if status:
            ax_grid.set_title(f"{base}\n{status}", fontsize=10)
        else:
            ax_grid.set_title(base, fontsize=11)

    def refresh_table_text() -> None:
        table_text.set_text(
            format_connection_table(
                ui_state["connections"],
                ui_state["copper_layers"],
                ui_state["route_status"],
            )
        )

    def clear_wires() -> None:
        for artist in wire_artists:
            artist.remove()
        wire_artists.clear()

    def clear_route_artists() -> None:
        for artist in route_artists:
            artist.remove()
        route_artists.clear()

    def draw_direct_wires(connections: List[Connection]) -> None:
        clear_wires()
        for conn in connections:
            ic_a, ic_b = conn.ic_a - 1, conn.ic_b - 1
            x1, y1 = pin_xy(centers[ic_a], conn.pin_a, offsets)
            x2, y2 = pin_xy(centers[ic_b], conn.pin_b, offsets)
            (line,) = ax_grid.plot(
                [x1, x2],
                [y1, y2],
                color="#bbbbbb",
                linewidth=1.0,
                alpha=0.45,
                linestyle="--",
                zorder=1,
            )
            wire_artists.append(line)

    def draw_committed_paths(committed: list) -> None:
        for poly, layer, conn_idx in committed:
            if len(poly) < 2:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            color = LAYER_TRACE_COLORS[layer % len(LAYER_TRACE_COLORS)]
            (line,) = ax_grid.plot(
                xs,
                ys,
                color=color,
                linewidth=2.4,
                alpha=0.9,
                zorder=3,
                solid_capstyle="round",
            )
            route_artists.append(line)
            route_artists.append(
                ax_grid.text(
                    poly[-1][0],
                    poly[-1][1],
                    f"L{layer + 1}",
                    fontsize=7,
                    color=color,
                    zorder=4,
                )
            )

    def render_route_frame(frame: RouteFrame) -> None:
        clear_route_artists()

        if frame.explored:
            pts = [(x + 0.5, y + 0.5) for x, y, _ in frame.explored]
            xs, ys = zip(*pts)
            sc = ax_grid.scatter(
                xs,
                ys,
                s=28,
                c="#f9e79f",
                edgecolors="#f4d03f",
                linewidths=0.3,
                alpha=0.55,
                zorder=2,
            )
            route_artists.append(sc)

        if frame.current is not None:
            cx, cy, cl = frame.current
            cur = ax_grid.scatter(
                [cx + 0.5],
                [cy + 0.5],
                s=90,
                c="#e74c3c",
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
                marker="s",
            )
            route_artists.append(cur)

        if frame.partial_path and len(frame.partial_path) >= 2 and frame.phase == "explore":
            xs = [p[0] for p in frame.partial_path]
            ys = [p[1] for p in frame.partial_path]
            (partial,) = ax_grid.plot(
                xs,
                ys,
                color="#7f8c8d",
                linewidth=1.5,
                linestyle=":",
                alpha=0.8,
                zorder=3,
            )
            route_artists.append(partial)

        draw_committed_paths(frame.committed)
        ui_state["route_status"] = frame.status
        refresh_table_text()
        set_title(frame.status)

    def stop_routing() -> None:
        nonlocal route_timer, route_gen
        if route_timer is not None:
            route_timer.stop()
            route_timer = None
        route_gen = None
        ui_state["routing"] = False
        btn_route.label.set_text("Router (pathfinder)")
        btn_regen.active = True
        btn_route.active = True

    def route_step() -> None:
        nonlocal route_gen
        if route_gen is None:
            return
        try:
            frame = next(route_gen)
        except StopIteration:
            stop_routing()
            fig.canvas.draw_idle()
            return
        render_route_frame(frame)
        if frame.phase == "done":
            stop_routing()
        fig.canvas.draw_idle()

    def start_routing(_event=None) -> None:
        nonlocal route_timer, route_gen
        if ui_state["routing"] or not ui_state["connections"]:
            return

        stop_routing()
        clear_route_artists()
        clear_wires()
        ui_state["route_status"] = "Routage en cours…"
        ui_state["routing"] = True
        btn_route.label.set_text("Routage…")
        btn_regen.active = False
        refresh_table_text()
        set_title("Initialisation A* multi-couches")

        route_gen = route_connections_iter(
            ui_state["connections"],
            centers,
            offsets,
            GRID_SIZE,
            ui_state["copper_layers"],
        )
        route_timer = fig.canvas.new_timer(interval=ROUTE_TIMER_MS)
        route_timer.add_callback(route_step)
        route_timer.start()
        route_step()

    def apply_connections(_event=None) -> None:
        stop_routing()
        clear_route_artists()
        ui_state["connections"] = random_connection_table(rng)
        ui_state["route_status"] = ""
        refresh_table_text()
        draw_direct_wires(ui_state["connections"])
        set_title()
        fig.canvas.draw_idle()

    def on_copper_submit(text: str) -> None:
        if ui_state["routing"]:
            copper_box.set_val(str(ui_state["copper_layers"]))
            return
        parsed = parse_copper_layers(text)
        if parsed is None:
            copper_box.set_val(str(ui_state["copper_layers"]))
            return
        ui_state["copper_layers"] = parsed
        refresh_table_text()
        fig.canvas.draw_idle()

    copper_box = TextBox(
        ax_copper,
        "",
        initial=str(DEFAULT_COPPER_LAYERS),
        textalignment="center",
    )
    copper_box.on_submit(on_copper_submit)

    btn_regen = Button(ax_btn_regen, "Regénérer la table")
    btn_regen.on_clicked(apply_connections)

    btn_route = Button(ax_btn_route, "Router (pathfinder)")
    btn_route.on_clicked(start_routing)

    set_title()
    apply_connections()
    plt.show()


if __name__ == "__main__":
    main()
