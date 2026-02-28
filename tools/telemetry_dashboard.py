#!/usr/bin/env python3
"""Simple telemetry dashboard for Gravity-Lang dump_all CSV output.

Usage:
  python tools/telemetry_dashboard.py artifacts/rocket_dump.csv --body Rocket
  python tools/telemetry_dashboard.py artifacts/rocket_dump.csv --body Rocket --live
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt

EARTH_RADIUS = 6_371_000.0


def read_body_rows(csv_path: Path, body: str):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("body") != body:
                continue
            x = float(r["x"])
            y = float(r["y"])
            z = float(r["z"])
            vx = float(r["vx"])
            vy = float(r["vy"])
            vz = float(r["vz"])
            altitude = max(0.0, math.sqrt(x * x + y * y + z * z) - EARTH_RADIUS)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            rows.append((int(r["step"]), altitude, speed))
    return rows


def draw(ax_alt, ax_vel, ax_traj, rows, body: str):
    ax_alt.clear()
    ax_vel.clear()
    ax_traj.clear()
    if not rows:
        ax_alt.set_title(f"No rows for body={body}")
        return

    steps = [r[0] for r in rows]
    alt_km = [r[1] / 1000.0 for r in rows]
    speed = [r[2] for r in rows]

    ax_alt.plot(steps, alt_km, color="#4cc9f0", linewidth=2.2)
    ax_alt.fill_between(steps, alt_km, color="#4cc9f0", alpha=0.2)
    ax_alt.set_title(f"{body} altitude profile", color="#e9ecef")
    ax_alt.set_ylabel("Altitude (km)")
    ax_alt.grid(True, alpha=0.2, linestyle="--")

    ax_vel.plot(steps, speed, color="#ff9f1c", linewidth=2.0)
    ax_vel.set_title(f"{body} speed", color="#e9ecef")
    ax_vel.set_ylabel("Speed (m/s)")
    ax_vel.set_xlabel("Step")
    ax_vel.grid(True, alpha=0.2, linestyle="--")

    # Use altitude vs speed as a compact phase-space style trajectory panel.
    ax_traj.plot(alt_km, speed, color="#80ed99", linewidth=1.8)
    ax_traj.set_title("Flight trajectory map (altitude vs speed)", color="#e9ecef")
    ax_traj.set_xlabel("Altitude (km)")
    ax_traj.set_ylabel("Speed (m/s)")
    ax_traj.grid(True, alpha=0.2, linestyle=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--body", default="Rocket")
    ap.add_argument("--live", action="store_true", help="Poll file every second")
    args = ap.parse_args()

    plt.style.use("dark_background")
    fig, (ax_alt, ax_vel, ax_traj) = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    fig.suptitle("Gravity-Lang Telemetry Dashboard", fontsize=14, color="#f8f9fa")

    while True:
        rows = read_body_rows(args.csv, args.body)
        draw(ax_alt, ax_vel, ax_traj, rows, args.body)
        plt.pause(0.1)
        if not args.live:
            break
        time.sleep(1.0)

    plt.show()


if __name__ == "__main__":
    main()
