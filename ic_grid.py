#!/usr/bin/env python3
"""Affiche 4 CI à 4 broches sur une grille et route les connexions par pathfinding."""

import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from layout import (
    DEFAULT_COPPER_LAYERS,
    GRID_SIZE,
    MAX_COPPER_LAYERS,
    MIN_CELL_GAP,
    MIN_COPPER_LAYERS,
    N_COMPONENTS,
    N_PINS,
    pin_corner_offsets,
    random_connection_table,
    random_ic_cells,
)
from pathfinder import Connection, RouteFrame, route_connections_iter

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




def parse_copper_layers(text: str) -> Optional[int]:
    try:
        n = int(text.strip())
    except ValueError:
        return None
    if MIN_COPPER_LAYERS <= n <= MAX_COPPER_LAYERS:
        return n
    return None


def format_connection_header(copper_layers: int, can_select: bool) -> str:
    header = (
        f"Couches de cuivre : {copper_layers}\n"
        "Table de connexions\n"
        + "─" * 22
    )
    if can_select:
        header += "\n(clic sur une ligne)"
    return header


def format_connection_row(conn: Connection, selected: bool) -> str:
    label = f"CI{conn.ic_a}-P{conn.pin_a}  ↔  CI{conn.ic_b}-P{conn.pin_b}"
    return f"▶ {label}" if selected else f"  {label}"


def pin_xy(center: tuple[int, int], pin: int, offsets: np.ndarray) -> tuple[float, float]:
    x, y = center
    dx, dy = offsets[pin - 1]
    return x + dx, y + dy


def main() -> None:
    rng = random.Random()
    colors = CONN_COLORS[:N_COMPONENTS]
    ic_cells = random_ic_cells(N_COMPONENTS, GRID_SIZE, MIN_CELL_GAP, rng)
    offsets = pin_corner_offsets()

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

    for i, (cx, cy) in enumerate(ic_cells):
        pins = offsets + np.array([cx, cy])
        ax_grid.add_patch(
            plt.Rectangle(
                (cx, cy),
                1,
                1,
                fill=True,
                facecolor=colors[i],
                alpha=0.12,
                edgecolor=colors[i],
                linewidth=1.2,
                zorder=3,
            )
        )
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
    TABLE_LINE_DY = 0.048
    table_header_text = ax_table.text(
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
    table_status_text = ax_table.text(
        0.05,
        0.08,
        "",
        transform=ax_table.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        family="monospace",
        color="#555555",
    )
    table_row_texts: list = []

    wire_artists: list = []
    route_artists: list = []
    route_timer = None
    route_gen = None

    ui_state = {
        "connections": [],
        "copper_layers": DEFAULT_COPPER_LAYERS,
        "route_status": "",
        "routing": False,
        "committed_paths": [],
        "selected_conn": None,
    }

    def set_title(status: str = "") -> None:
        base = "Grille PCB — composants et routage"
        if status:
            ax_grid.set_title(f"{base}\n{status}", fontsize=10)
        else:
            ax_grid.set_title(base, fontsize=11)

    def clear_table_rows() -> None:
        for artist in table_row_texts:
            artist.remove()
        table_row_texts.clear()

    def refresh_table_text() -> None:
        connections = ui_state["connections"]
        can_select = bool(ui_state["committed_paths"]) and not ui_state["routing"]
        header_lines = 3 + (1 if can_select else 0)
        table_header_text.set_text(
            format_connection_header(ui_state["copper_layers"], can_select)
        )
        table_status_text.set_text(ui_state["route_status"])
        clear_table_rows()
        y = 0.95 - header_lines * TABLE_LINE_DY
        for idx, conn in enumerate(connections):
            selected = idx == ui_state["selected_conn"]
            row_style = {
                "color": "#1a5276",
                "weight": "bold",
                "bbox": dict(boxstyle="round,pad=0.25", facecolor="#d6eaf8", edgecolor="#2980b9"),
            } if selected else {"color": "#333333", "weight": "normal"}
            row = ax_table.text(
                0.05,
                y,
                format_connection_row(conn, selected),
                transform=ax_table.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
                picker=5 if can_select else False,
                **row_style,
            )
            row._conn_index = idx  # type: ignore[attr-defined]
            table_row_texts.append(row)
            y -= TABLE_LINE_DY

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
            x1, y1 = pin_xy(ic_cells[ic_a], conn.pin_a, offsets)
            x2, y2 = pin_xy(ic_cells[ic_b], conn.pin_b, offsets)
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

    def draw_committed_paths(committed: list, highlight_idx: Optional[int] = None) -> None:
        has_highlight = highlight_idx is not None and any(
            entry[2] == highlight_idx for entry in committed
        )
        for entry in committed:
            poly, layer, conn_idx = entry[0], entry[1], entry[2]
            vias = entry[3] if len(entry) > 3 else []
            if len(poly) < 2:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            color = LAYER_TRACE_COLORS[layer % len(LAYER_TRACE_COLORS)]
            highlighted = has_highlight and conn_idx == highlight_idx
            dimmed = has_highlight and not highlighted
            (line,) = ax_grid.plot(
                xs,
                ys,
                color=color,
                linewidth=4.2 if highlighted else 2.4,
                alpha=1.0 if highlighted else (0.22 if dimmed else 0.9),
                zorder=5 if highlighted else 3,
                solid_capstyle="round",
            )
            route_artists.append(line)
            if vias and not dimmed:
                vx, vy = zip(*vias)
                route_artists.append(
                    ax_grid.scatter(
                        vx,
                        vy,
                        s=55,
                        marker="D",
                        c="#f1c40f",
                        edgecolors="#b7950b",
                        linewidths=0.8,
                        zorder=6,
                    )
                )
            if not dimmed:
                route_artists.append(
                    ax_grid.text(
                        poly[-1][0],
                        poly[-1][1],
                        f"L{layer + 1}",
                        fontsize=7 if not highlighted else 8,
                        color=color,
                        weight="bold" if highlighted else "normal",
                        zorder=6 if highlighted else 4,
                    )
                )
            if highlighted:
                route_artists.append(
                    ax_grid.scatter(
                        [xs[0], xs[-1]],
                        [ys[0], ys[-1]],
                        s=140,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=2.5,
                        zorder=7,
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

        draw_committed_paths(frame.committed, ui_state["selected_conn"])
        ui_state["route_status"] = frame.status
        if frame.phase == "done":
            ui_state["committed_paths"] = list(frame.committed)
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
        ui_state["committed_paths"] = []
        ui_state["selected_conn"] = None
        ui_state["route_status"] = "Routage en cours…"
        ui_state["routing"] = True
        btn_route.label.set_text("Routage…")
        btn_regen.active = False
        refresh_table_text()
        set_title("Initialisation A* multi-couches")

        route_gen = route_connections_iter(
            ui_state["connections"],
            ic_cells,
            offsets,
            GRID_SIZE,
            ui_state["copper_layers"],
        )
        route_timer = fig.canvas.new_timer(interval=ROUTE_TIMER_MS)
        route_timer.add_callback(route_step)
        route_timer.start()
        route_step()

    def redraw_committed_routes() -> None:
        clear_route_artists()
        if ui_state["committed_paths"]:
            draw_committed_paths(
                ui_state["committed_paths"],
                ui_state["selected_conn"],
            )

    def on_connection_pick(event) -> None:
        if ui_state["routing"] or not ui_state["committed_paths"]:
            return
        artist = event.artist
        if artist not in table_row_texts:
            return
        idx = getattr(artist, "_conn_index", None)
        if idx is None:
            return
        ui_state["selected_conn"] = None if ui_state["selected_conn"] == idx else idx
        refresh_table_text()
        redraw_committed_routes()
        fig.canvas.draw_idle()

    def apply_connections(_event=None) -> None:
        stop_routing()
        clear_route_artists()
        ui_state["connections"] = random_connection_table(rng)
        ui_state["route_status"] = ""
        ui_state["committed_paths"] = []
        ui_state["selected_conn"] = None
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

    fig.canvas.mpl_connect("pick_event", on_connection_pick)

    set_title()
    apply_connections()
    plt.show()


if __name__ == "__main__":
    main()
