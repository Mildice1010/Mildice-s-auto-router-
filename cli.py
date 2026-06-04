#!/usr/bin/env python3
"""CLI du routeur PCB — routage headless et simulations parallèles."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from layout import (
    DEFAULT_COPPER_LAYERS,
    GRID_SIZE,
    MAX_COPPER_LAYERS,
    MIN_COPPER_LAYERS,
)
from simulation import RoutingResult, run_routing, run_routing_task

DEFAULT_SUMMARY_CSV = "simulation_runs.csv"
SUMMARY_CSV_COLUMNS = [
    "timestamp_utc",
    "n_simulations",
    "base_seed",
    "grid_size",
    "copper_layers",
    "jobs",
    "avg_success_rate_pct",
]


def _print_result(result: RoutingResult, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    print(
        f"{prefix}{result.succeeded} connexion(s) réussie(s), "
        f"{result.failed} ratée(s) / {result.total_connections} "
        f"(rip-up: {result.rip_events})"
    )


def _print_summary(results: List[RoutingResult]) -> None:
    total_ok = sum(r.succeeded for r in results)
    total_fail = sum(r.failed for r in results)
    total_conns = sum(r.total_connections for r in results)
    total_rips = sum(r.rip_events for r in results)
    n = len(results)

    print()
    print("─── Bilan ───")
    print(f"Simulations : {n}")
    print(f"Connexions réussies : {total_ok}")
    print(f"Connexions ratées   : {total_fail}")
    print(f"Total connexions    : {total_conns}")
    if total_conns:
        rate = 100.0 * total_ok / total_conns
        print(f"Taux de succès      : {rate:.1f} %")
    print(f"Événements rip-up   : {total_rips}")


def _mean_success_rate_pct(results: List[RoutingResult]) -> float:
    """Moyenne des taux de succès de chaque simulation du lot."""
    rates = [
        100.0 * r.success_rate
        for r in results
        if r.total_connections > 0
    ]
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def _append_run_summary_csv(
    path: Path,
    *,
    results: List[RoutingResult],
    n_simulations: int,
    base_seed: Optional[int],
    grid_size: int,
    copper_layers: int,
    jobs: int,
) -> None:
    """Ajoute une ligne (paramètres + taux de succès moyen) pour un lot de simulations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = [
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_simulations,
        "" if base_seed is None else base_seed,
        grid_size,
        copper_layers,
        jobs,
        f"{_mean_success_rate_pct(results):.2f}",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(SUMMARY_CSV_COLUMNS)
        writer.writerow(row)


def cmd_route(args: argparse.Namespace) -> int:
    result = run_routing(
        seed=args.seed,
        grid_size=args.grid_size,
        copper_layers=args.layers,
    )
    _print_result(result)
    return 0 if result.failed == 0 else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    n = args.n
    workers = args.jobs or min(n, os.cpu_count() or 1)
    base_seed = args.seed

    tasks = []
    for i in range(n):
        seed = None if base_seed is None else base_seed + i
        tasks.append(
            {
                "seed": seed,
                "grid_size": args.grid_size,
                "copper_layers": args.layers,
            }
        )

    print(f"Lancement de {n} simulation(s) en parallèle ({workers} worker(s))…")
    indexed: List[Tuple[int, RoutingResult]] = []
    verbose = not args.quiet

    if workers <= 1:
        for i, params in enumerate(tasks, start=1):
            result = run_routing(**params)
            indexed.append((i, result))
            if verbose:
                seed_label = params["seed"] if params["seed"] is not None else "aléatoire"
                _print_result(result, f"  [{i}/{n}] seed={seed_label}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_routing_task, params): (i, params)
                for i, params in enumerate(tasks, start=1)
            }
            for future in as_completed(futures):
                i, params = futures[future]
                result = future.result()
                indexed.append((i, result))
                if verbose:
                    seed_label = params["seed"] if params["seed"] is not None else "aléatoire"
                    _print_result(result, f"  [{i}/{n}] seed={seed_label}")

    indexed.sort(key=lambda x: x[0])
    results = [r for _, r in indexed]

    if not args.no_csv:
        out = Path(args.csv)
        _append_run_summary_csv(
            out,
            results=results,
            n_simulations=n,
            base_seed=base_seed,
            grid_size=args.grid_size,
            copper_layers=args.layers,
            jobs=workers,
        )
        print(f"\nLot enregistré (moyenne) : {out.resolve()}")

    _print_summary(results)
    any_fail = any(r.failed > 0 for r in results)
    return 1 if any_fail else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire (pour simulate : seed + i par run)",
    )
    common.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help=f"Taille de la grille (défaut: {GRID_SIZE})",
    )
    common.add_argument(
        "--layers",
        type=int,
        default=DEFAULT_COPPER_LAYERS,
        help=f"Couches de cuivre (défaut: {DEFAULT_COPPER_LAYERS})",
    )

    parser = argparse.ArgumentParser(
        description="Routeur PCB Mildice — simulations en ligne de commande.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s route
  %(prog)s route --seed 42 --layers 4
  %(prog)s simulate -n 8
  %(prog)s simulate -n 16 --jobs 8 --seed 1000
  %(prog)s simulate -n 50 --seed 0 -q
  %(prog)s simulate -n 100 -o experiments.csv --layers 4
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser(
        "route",
        parents=[common],
        help="Une simulation (placement + table + routage)",
    )
    p_route.set_defaults(func=cmd_route)

    p_sim = sub.add_parser(
        "simulate",
        parents=[common],
        help="N simulations en parallèle",
    )
    p_sim.add_argument(
        "-n",
        "--count",
        dest="n",
        type=int,
        required=True,
        metavar="N",
        help="Nombre de simulations à lancer",
    )
    p_sim.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="W",
        help="Workers parallèles (défaut: min(N, CPU))",
    )
    p_sim.add_argument(
        "-o",
        "--csv",
        type=str,
        default=DEFAULT_SUMMARY_CSV,
        metavar="FICHIER",
        help=f"CSV des lots (paramètres + taux moyen), défaut: {DEFAULT_SUMMARY_CSV}",
    )
    p_sim.add_argument(
        "--no-csv",
        action="store_true",
        help="Ne pas enregistrer dans le CSV de synthèse",
    )
    p_sim.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Ne pas afficher le détail de chaque simulation",
    )
    p_sim.set_defaults(func=cmd_simulate)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (MIN_COPPER_LAYERS <= args.layers <= MAX_COPPER_LAYERS):
        print(
            f"Erreur: --layers doit être entre {MIN_COPPER_LAYERS} et {MAX_COPPER_LAYERS}.",
            file=sys.stderr,
        )
        return 2
    if args.grid_size < 6:
        print("Erreur: --grid-size doit être >= 6.", file=sys.stderr)
        return 2
    if hasattr(args, "n") and args.n < 1:
        print("Erreur: -n doit être >= 1.", file=sys.stderr)
        return 2

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
