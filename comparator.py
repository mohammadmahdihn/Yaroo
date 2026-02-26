"""
MACTER vs Random Baseline — Comparison & Visualization
=======================================================
Runs both algorithms across several experiment axes (matching the paper's
evaluation style) and produces a multi-panel figure.

Experiment axes
---------------
1. Varying number of vehicles  (Fig 5 / Fig 6 style)
2. Varying task data size       (Fig 4 style)
3. Varying max tolerable delay  (Fig 8 style)
4. Per-vehicle utility scatter  (single snapshot, N=30)

Baseline algorithms compared
-----------------------------
  MACTER  – the fixed distributed MACTER algorithm (Algorithm 2)
  Random  – each vehicle independently picks loc/vec with prob=0.5;
             offloaders receive equal-share CPU (F_total / num_offloaders)
  All-Local  – all vehicles always compute locally (lower bound)
  All-VEC    – all vehicles always offload; equal-share CPU (upper-bound reference)
"""

from __future__ import annotations

import copy
import math
import random
import sys
import os
import statistics
from typing import Dict, List, Tuple, Optional

# ── import the fixed implementation ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from new_money import (
    SystemParams, Vehicle, Task,
    utility_local, utility_vec,
    allocate_edge_cpu_bisection,
    distributed_macter,
    make_vehicles, group_by_rsu,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Baseline algorithms
# ══════════════════════════════════════════════════════════════════════════════

def random_offload(vehicles: List[Vehicle], rsu_F: float,
                   seed: Optional[int] = None) -> Tuple[Dict[int, float], Dict[int, str], float]:
    """Each vehicle flips a fair coin: 50 % VEC, 50 % local.
    VEC vehicles receive equal CPU shares (F_total / #offloaders)."""
    rng = random.Random(seed)
    decisions: Dict[int, str] = {}

    for v in vehicles:
        if v.rate_bps <= 0.0:          # can't offload without connectivity
            decisions[v.vid] = "loc"
        else:
            decisions[v.vid] = "vec" if rng.random() < 0.5 else "loc"

    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    n_off = len(offloaders)

    alloc: Dict[int, float] = {}
    if n_off > 0:
        f_each = rsu_F / n_off
        for v in offloaders:
            alloc[v.vid] = f_each

    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)


def all_local(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], float]:
    """Lower bound: every vehicle computes locally."""
    decisions = {v.vid: "loc" for v in vehicles}
    alloc: Dict[int, float] = {}
    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)


def all_vec(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], float]:
    """Reference: all vehicles offload, equal CPU share."""
    eligible = [v for v in vehicles if v.rate_bps > 0.0]
    ineligible = [v for v in vehicles if v.rate_bps <= 0.0]

    decisions: Dict[int, str] = {}
    for v in eligible:
        decisions[v.vid] = "vec"
    for v in ineligible:
        decisions[v.vid] = "loc"

    n = len(eligible)
    alloc: Dict[int, float] = {}
    if n > 0:
        f_each = rsu_F / n
        for v in eligible:
            alloc[v.vid] = f_each

    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)


# ══════════════════════════════════════════════════════════════════════════════
# Shared metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_efficiency(vehicles: List[Vehicle],
                        alloc: Dict[int, float],
                        decisions: Dict[int, str]) -> float:
    total_u, total_e = 0.0, 0.0
    for v in vehicles:
        if decisions[v.vid] == "vec":
            f_v = alloc.get(v.vid, 0.0)
            u = utility_vec(v, f_v) if f_v > 0.0 else float("-inf")
            e = v.local_energy_J() + v.vec_energy_J()
        else:
            u = utility_local(v)
            e = v.local_energy_J()
        total_u += max(u, 0.0)
        total_e += max(e, 1e-12)
    return total_u / total_e if total_e > 0 else 0.0


def _avg_utility(vehicles: List[Vehicle],
                 alloc: Dict[int, float],
                 decisions: Dict[int, str]) -> float:
    vals = []
    for v in vehicles:
        if decisions[v.vid] == "vec":
            f_v = alloc.get(v.vid, 0.0)
            u = utility_vec(v, f_v) if f_v > 0.0 else 0.0
        else:
            u = utility_local(v)
        vals.append(max(u, 0.0))
    return statistics.mean(vals) if vals else 0.0


def _vec_ratio(decisions: Dict[int, str]) -> float:
    vals = list(decisions.values())
    return vals.count("vec") / len(vals) if vals else 0.0


def _avg_delay(vehicles: List[Vehicle],
               alloc: Dict[int, float],
               decisions: Dict[int, str]) -> float:
    vals = []
    for v in vehicles:
        if decisions[v.vid] == "vec":
            f_v = alloc.get(v.vid, 0.0)
            t = v.vec_time(f_v) if f_v > 0.0 else float("inf")
        else:
            t = v.local_time()
        if math.isfinite(t):
            vals.append(t)
    return statistics.mean(vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Run helpers: sweep over one axis, repeat trials, average
# ══════════════════════════════════════════════════════════════════════════════

TRIALS = 10  # Monte-Carlo trials per data point


def _run_one_scenario(vehicles: List[Vehicle], params: SystemParams) -> dict:
    """Run all four algorithms on a pre-built vehicle list; return metric dict."""
    groups = group_by_rsu(vehicles, params.M)

    # ── MACTER ────────────────────────────────────────────────────────────────
    m_eff, m_util, m_vratio, m_delay = [], [], [], []
    for vs in groups.values():
        if not vs:
            continue
        alloc, dec, eff = distributed_macter(vs, params.F_vec_total_GHz,
                                             params.max_iter_algo2)
        m_eff.append(eff)
        m_util.append(_avg_utility(vs, alloc, dec))
        m_vratio.append(_vec_ratio(dec))
        m_delay.append(_avg_delay(vs, alloc, dec))

    # ── Random (average over several coin-flip seeds) ─────────────────────────
    r_eff_t, r_util_t, r_vratio_t, r_delay_t = [], [], [], []
    for trial in range(TRIALS):
        t_eff, t_util, t_vr, t_delay = [], [], [], []
        for vs in groups.values():
            if not vs:
                continue
            alloc, dec, eff = random_offload(vs, params.F_vec_total_GHz, seed=trial)
            t_eff.append(eff)
            t_util.append(_avg_utility(vs, alloc, dec))
            t_vr.append(_vec_ratio(dec))
            t_delay.append(_avg_delay(vs, alloc, dec))
        r_eff_t.append(_mean(t_eff))
        r_util_t.append(_mean(t_util))
        r_vratio_t.append(_mean(t_vr))
        r_delay_t.append(_mean(t_delay))

    # ── All-local ─────────────────────────────────────────────────────────────
    l_eff, l_util, l_delay = [], [], []
    for vs in groups.values():
        if not vs:
            continue
        alloc, dec, eff = all_local(vs, params.F_vec_total_GHz)
        l_eff.append(eff)
        l_util.append(_avg_utility(vs, alloc, dec))
        l_delay.append(_avg_delay(vs, alloc, dec))

    # ── All-VEC ───────────────────────────────────────────────────────────────
    v_eff, v_util, v_delay = [], [], []
    for vs in groups.values():
        if not vs:
            continue
        alloc, dec, eff = all_vec(vs, params.F_vec_total_GHz)
        v_eff.append(eff)
        v_util.append(_avg_utility(vs, alloc, dec))
        v_delay.append(_avg_delay(vs, alloc, dec))

    return dict(
        macter_eff=_mean(m_eff),       macter_util=_mean(m_util),
        macter_vr=_mean(m_vratio),     macter_delay=_mean(m_delay),

        rand_eff=_mean(r_eff_t),       rand_util=_mean(r_util_t),
        rand_vr=_mean(r_vratio_t),     rand_delay=_mean(r_delay_t),
        rand_eff_std=_std(r_eff_t),    rand_util_std=_std(r_util_t),

        local_eff=_mean(l_eff),        local_util=_mean(l_util),
        local_delay=_mean(l_delay),

        allvec_eff=_mean(v_eff),       allvec_util=_mean(v_util),
        allvec_delay=_mean(v_delay),
    )


def _mean(lst):
    return statistics.mean(lst) if lst else 0.0

def _std(lst):
    return statistics.stdev(lst) if len(lst) > 1 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Sweep: varying number of vehicles
# ══════════════════════════════════════════════════════════════════════════════

def sweep_num_vehicles(N_vals, params: SystemParams, outer_trials: int = 8):
    results = {k: [] for k in [
        "macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
        "macter_util","rand_util","rand_util_std","local_util","allvec_util",
        "macter_vr","rand_vr","macter_delay","rand_delay","local_delay","allvec_delay",
    ]}
    for N in N_vals:
        print(f"  sweep_N: N={N}", flush=True)
        # Average over outer_trials independent vehicle placements
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, params, seed=t * 1000 + N)
            trial_data.append(_run_one_scenario(vehicles, params))

        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                # std across outer trials of the mean metric
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Sweep: varying data size
# ══════════════════════════════════════════════════════════════════════════════

def sweep_data_size(alpha_vals_kB, params: SystemParams, N: int = 20, outer_trials: int = 8):
    results = {k: [] for k in [
        "macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
    ]}
    for a in alpha_vals_kB:
        print(f"  sweep_alpha: alpha={a}kB", flush=True)
        p = copy.copy(params)
        p.alpha_kB_min = a * 0.85
        p.alpha_kB_max = a * 1.15
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, p, seed=t * 500 + int(a))
            trial_data.append(_run_one_scenario(vehicles, p))
        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Sweep: varying max tolerable delay
# ══════════════════════════════════════════════════════════════════════════════

def sweep_tmax(tmax_vals, params: SystemParams, N: int = 20, outer_trials: int = 8):
    results = {k: [] for k in [
        "macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
        "macter_util","rand_util","rand_util_std","local_util","allvec_util",
    ]}
    for tmax in tmax_vals:
        print(f"  sweep_tmax: tmax={tmax:.1f}s", flush=True)
        p = copy.copy(params)
        p.tmax_min = tmax * 0.7
        p.tmax_max = tmax * 1.3
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, p, seed=t * 300 + int(tmax * 100))
            trial_data.append(_run_one_scenario(vehicles, p))
        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Per-vehicle snapshot
# ══════════════════════════════════════════════════════════════════════════════

def per_vehicle_snapshot(vehicles: List[Vehicle], params: SystemParams):
    groups = group_by_rsu(vehicles, params.M)

    macter_u: Dict[int, float] = {}
    rand_u:   Dict[int, float] = {}
    macter_dec: Dict[int, str] = {}
    rand_dec:   Dict[int, str] = {}

    for vs in groups.values():
        if not vs:
            continue
        alloc_m, dec_m, _ = distributed_macter(vs, params.F_vec_total_GHz,
                                               params.max_iter_algo2)
        # random: average 20 trials per vehicle
        rand_u_trials: Dict[int, List[float]] = {v.vid: [] for v in vs}
        rand_dec_majority: Dict[int, Dict[str, int]] = {v.vid: {"vec": 0, "loc": 0} for v in vs}

        for trial in range(20):
            alloc_r, dec_r, _ = random_offload(vs, params.F_vec_total_GHz, seed=trial)
            for v in vs:
                if dec_r[v.vid] == "vec":
                    u = utility_vec(v, alloc_r.get(v.vid, 0.0))
                else:
                    u = utility_local(v)
                rand_u_trials[v.vid].append(max(u, 0.0))
                rand_dec_majority[v.vid][dec_r[v.vid]] += 1

        for v in vs:
            # MACTER utility
            if dec_m[v.vid] == "vec":
                u = utility_vec(v, alloc_m.get(v.vid, 0.0))
            else:
                u = utility_local(v)
            macter_u[v.vid] = max(u, 0.0)
            macter_dec[v.vid] = dec_m[v.vid]

            rand_u[v.vid] = _mean(rand_u_trials[v.vid])
            rand_dec[v.vid] = "vec" if rand_dec_majority[v.vid]["vec"] >= 10 else "loc"

    return macter_u, rand_u, macter_dec, rand_dec


# ══════════════════════════════════════════════════════════════════════════════
# Styling helpers
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "macter":  "#1a6faf",
    "random":  "#e05c2a",
    "local":   "#4caa4c",
    "allvec":  "#9b59b6",
}

MARKERS = {"macter": "o", "random": "s", "local": "^", "allvec": "D"}
LWIDTH  = 2.0
MSIZE   = 7


def _plot_line(ax, x, y, yerr, key, label, linestyle="-", alpha_fill=0.12):
    c = COLORS[key]
    m = MARKERS[key]
    ax.plot(x, y, color=c, marker=m, markersize=MSIZE, linewidth=LWIDTH,
            linestyle=linestyle, label=label, zorder=3)
    if yerr is not None:
        y_arr = np.array(y)
        e_arr = np.array(yerr)
        ax.fill_between(x, y_arr - e_arr, y_arr + e_arr,
                        color=c, alpha=alpha_fill, zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
# Build the figure
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(out_path: str):
    params = SystemParams()

    # ── Experiment 1: vary N ──────────────────────────────────────────────────
    N_vals = [5, 10, 15, 20, 25, 30, 40, 50]
    print("Running sweep: num_vehicles …")
    res_N = sweep_num_vehicles(N_vals, params, outer_trials=8)

    # ── Experiment 2: vary data size ──────────────────────────────────────────
    alpha_vals = [10, 20, 30, 40, 50, 60, 70, 80]
    print("Running sweep: data size …")
    res_alpha = sweep_data_size(alpha_vals, params, N=20, outer_trials=8)

    # ── Experiment 3: vary tmax ───────────────────────────────────────────────
    tmax_vals = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    print("Running sweep: max tolerable delay …")
    res_tmax = sweep_tmax(tmax_vals, params, N=20, outer_trials=8)

    # ── Experiment 4: per-vehicle snapshot ───────────────────────────────────
    print("Running per-vehicle snapshot …")
    vehicles_snap = make_vehicles(30, params, seed=42)
    macter_u, rand_u, macter_dec, rand_dec = per_vehicle_snapshot(vehicles_snap, params)
    vids = sorted(macter_u.keys())
    snap_m = [macter_u[i] for i in vids]
    snap_r = [rand_u[i]   for i in vids]

    # ── Experiment 5: convergence (utility vs iteration) ─────────────────────
    print("Running convergence trace …")
    conv_data = _convergence_trace(params, N=25, seed=7)

    # ══════════════════════════════════════════════════════════════════════════
    # Layout: 3 rows × 2 cols + wide convergence panel + per-vehicle bar
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 20))
    fig.patch.set_facecolor("#f7f9fc")

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.52, wspace=0.35,
                           top=0.93, bottom=0.05,
                           left=0.07, right=0.97)

    ax = [
        fig.add_subplot(gs[0, 0]),   # 0 – efficiency vs N
        fig.add_subplot(gs[0, 1]),   # 1 – avg utility vs N
        fig.add_subplot(gs[1, 0]),   # 2 – efficiency vs data size
        fig.add_subplot(gs[1, 1]),   # 3 – efficiency vs tmax
        # fig.add_subplot(gs[2, 0]),   # 4 – avg utility vs tmax
        # fig.add_subplot(gs[2, 1]),   # 5 – VEC offload ratio vs N
        # fig.add_subplot(gs[3, 0]),   # 6 – convergence
        # fig.add_subplot(gs[3, 1]),   # 7 – per-vehicle utility bar
    ]

    for a in ax:
        a.set_facecolor("#ffffff")
        a.grid(True, linestyle="--", linewidth=0.55, alpha=0.55, color="#cccccc")
        a.spines[["top", "right"]].set_visible(False)

    # ── Panel 0: Computation Efficiency vs N ─────────────────────────────────
    _plot_line(ax[0], N_vals, res_N["macter_eff"],  res_N["rand_eff_std"],
               "macter", "MACTER")
    _plot_line(ax[0], N_vals, res_N["rand_eff"],    res_N["rand_eff_std"],
               "random", "Random")
    _plot_line(ax[0], N_vals, res_N["local_eff"],   None,
               "local",  "All-Local", linestyle="--")
    _plot_line(ax[0], N_vals, res_N["allvec_eff"],  None,
               "allvec", "All-VEC",   linestyle=":")
    ax[0].set_title("Computation Efficiency vs. Number of Vehicles",
                    fontweight="bold", fontsize=11)
    ax[0].set_xlabel("Number of Vehicles (N)")
    ax[0].set_ylabel("Computation Efficiency")
    ax[0].legend(fontsize=8.5, framealpha=0.7)

    # ── Panel 1: Avg Utility vs N ─────────────────────────────────────────────
    _plot_line(ax[1], N_vals, res_N["macter_util"], res_N["rand_util_std"],
               "macter", "MACTER")
    _plot_line(ax[1], N_vals, res_N["rand_util"],   res_N["rand_util_std"],
               "random", "Random")
    _plot_line(ax[1], N_vals, res_N["local_util"],  None,
               "local",  "All-Local", linestyle="--")
    _plot_line(ax[1], N_vals, res_N["allvec_util"], None,
               "allvec", "All-VEC",   linestyle=":")
    ax[1].set_title("Average Vehicle Utility vs. Number of Vehicles",
                    fontweight="bold", fontsize=11)
    ax[1].set_xlabel("Number of Vehicles (N)")
    ax[1].set_ylabel("Average Utility")
    ax[1].legend(fontsize=8.5, framealpha=0.7)

    # ── Panel 2: Efficiency vs Data Size ─────────────────────────────────────
    _plot_line(ax[2], alpha_vals, res_alpha["macter_eff"], None,
               "macter", "MACTER")
    _plot_line(ax[2], alpha_vals, res_alpha["rand_eff"],   res_alpha["rand_eff_std"],
               "random", "Random")
    _plot_line(ax[2], alpha_vals, res_alpha["local_eff"],  None,
               "local",  "All-Local", linestyle="--")
    _plot_line(ax[2], alpha_vals, res_alpha["allvec_eff"], None,
               "allvec", "All-VEC",   linestyle=":")
    ax[2].set_title("Computation Efficiency vs. Task Data Size",
                    fontweight="bold", fontsize=11)
    ax[2].set_xlabel("Task Data Size (kB)")
    ax[2].set_ylabel("Computation Efficiency")
    ax[2].legend(fontsize=8.5, framealpha=0.7)

    # ── Panel 3: Efficiency vs tmax ───────────────────────────────────────────
    _plot_line(ax[3], tmax_vals, res_tmax["macter_eff"], None,
               "macter", "MACTER")
    _plot_line(ax[3], tmax_vals, res_tmax["rand_eff"],   res_tmax["rand_eff_std"],
               "random", "Random")
    _plot_line(ax[3], tmax_vals, res_tmax["local_eff"],  None,
               "local",  "All-Local", linestyle="--")
    _plot_line(ax[3], tmax_vals, res_tmax["allvec_eff"], None,
               "allvec", "All-VEC",   linestyle=":")
    ax[3].set_title("Computation Efficiency vs. Max Tolerable Delay",
                    fontweight="bold", fontsize=11)
    ax[3].set_xlabel("Max Tolerable Delay t_max (s)")
    ax[3].set_ylabel("Computation Efficiency")
    ax[3].legend(fontsize=8.5, framealpha=0.7)

    # # ── Panel 4: Avg Utility vs tmax ──────────────────────────────────────────
    # _plot_line(ax[4], tmax_vals, res_tmax["macter_util"], None,
    #            "macter", "MACTER")
    # _plot_line(ax[4], tmax_vals, res_tmax["rand_util"],   res_tmax["rand_util_std"],
    #            "random", "Random")
    # _plot_line(ax[4], tmax_vals, res_tmax["local_util"],  None,
    #            "local",  "All-Local", linestyle="--")
    # _plot_line(ax[4], tmax_vals, res_tmax["allvec_util"], None,
    #            "allvec", "All-VEC",   linestyle=":")
    # ax[4].set_title("Average Vehicle Utility vs. Max Tolerable Delay",
    #                 fontweight="bold", fontsize=11)
    # ax[4].set_xlabel("Max Tolerable Delay t_max (s)")
    # ax[4].set_ylabel("Average Utility")
    # ax[4].legend(fontsize=8.5, framealpha=0.7)
    #
    # # ── Panel 5: VEC Offload Ratio vs N ──────────────────────────────────────
    # ax[5].plot(N_vals, [v * 100 for v in res_N["macter_vr"]],
    #            color=COLORS["macter"], marker=MARKERS["macter"],
    #            markersize=MSIZE, linewidth=LWIDTH, label="MACTER")
    # ax[5].axhline(50, color=COLORS["random"], linestyle="--", linewidth=1.6,
    #               label="Random (50 % by design)")
    # ax[5].set_title("VEC Offload Ratio vs. Number of Vehicles",
    #                 fontweight="bold", fontsize=11)
    # ax[5].set_xlabel("Number of Vehicles (N)")
    # ax[5].set_ylabel("VEC Offload Ratio (%)")
    # ax[5].set_ylim(0, 105)
    # ax[5].legend(fontsize=8.5, framealpha=0.7)
    #
    # # ── Panel 6: Convergence trace ────────────────────────────────────────────
    # iters_m, util_m = conv_data["macter_iters"], conv_data["macter_utils"]
    # iters_r, util_r = conv_data["rand_iters"],   conv_data["rand_utils"]
    #
    # ax[6].plot(iters_m, util_m, color=COLORS["macter"], marker="o",
    #            markersize=5, linewidth=LWIDTH, label="MACTER")
    # ax[6].axhline(util_r, color=COLORS["random"], linestyle="--",
    #               linewidth=1.8, label=f"Random baseline ({util_r:.3f})")
    # ax[6].axhline(conv_data["local_util"], color=COLORS["local"],
    #               linestyle=":", linewidth=1.8,
    #               label=f"All-Local ({conv_data['local_util']:.3f})")
    # ax[6].set_title("MACTER Convergence: Average Utility per Iteration",
    #                 fontweight="bold", fontsize=11)
    # ax[6].set_xlabel("Iteration")
    # ax[6].set_ylabel("Average Utility")
    # ax[6].legend(fontsize=8.5, framealpha=0.7)
    #
    # # ── Panel 7: Per-vehicle utility bar chart ────────────────────────────────
    # x = np.arange(len(vids))
    # width = 0.38
    # ax[7].bar(x - width / 2, snap_m, width, color=COLORS["macter"],
    #           alpha=0.82, label="MACTER", edgecolor="white", linewidth=0.4)
    # ax[7].bar(x + width / 2, snap_r, width, color=COLORS["random"],
    #           alpha=0.82, label="Random (avg)", edgecolor="white", linewidth=0.4)
    #
    # # colour the x-tick labels by MACTER decision
    # ax[7].set_xticks(x)
    # ax[7].set_xticklabels([f"v{i}" for i in vids], rotation=90, fontsize=6.5)
    # for tick, vid in zip(ax[7].get_xticklabels(), vids):
    #     tick.set_color(COLORS["macter"] if macter_dec.get(vid) == "vec"
    #                    else COLORS["local"])
    #
    # ax[7].set_title("Per-Vehicle Utility: MACTER vs Random  (N=30, seed=42)\n"
    #                 r"  $\bf{blue\ label}$=MACTER VEC,  $\bf{green\ label}$=MACTER Local",
    #                 fontweight="bold", fontsize=10)
    # ax[7].set_ylabel("Utility (clamped ≥ 0)")
    # leg_handles = [
    #     Line2D([0], [0], color=COLORS["macter"], lw=6, alpha=0.7, label="MACTER"),
    #     Line2D([0], [0], color=COLORS["random"],  lw=6, alpha=0.7, label="Random (avg)"),
    # ]
    # ax[7].legend(handles=leg_handles, fontsize=8.5, framealpha=0.7)

    # ── Compute overall improvement stats for super-title ────────────────────
    m_mean_eff = _mean(res_N["macter_eff"])
    r_mean_eff = _mean(res_N["rand_eff"])
    l_mean_eff = _mean(res_N["local_eff"])
    pct_vs_rand  = 100.0 * (m_mean_eff - r_mean_eff)  / max(r_mean_eff,  1e-12)
    pct_vs_local = 100.0 * (m_mean_eff - l_mean_eff)  / max(l_mean_eff, 1e-12)

    fig.suptitle(
        "MACTER vs Random Baseline — Computation Efficiency & Utility Comparison\n"
        f"MACTER improves efficiency by  +{pct_vs_rand:.0f}%  vs Random  "
        f"and  +{pct_vs_local:.0f}%  vs All-Local  (averaged across N sweep)",
        fontsize=13, fontweight="bold", color="#1a1a2e", y=0.975,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nFigure saved → {out_path}")
    return pct_vs_rand, pct_vs_local


# ══════════════════════════════════════════════════════════════════════════════
# Convergence trace helper
# ══════════════════════════════════════════════════════════════════════════════

def _convergence_trace(params: SystemParams, N: int = 25, seed: int = 7) -> dict:
    """Replicate Algorithm 2 step-by-step and record avg utility at each iteration."""
    from new_money import (
        algorithm1_resource_allocation_and_choice,
        allocate_edge_cpu_bisection,
    )

    vehicles = make_vehicles(N, params, seed=seed)
    groups   = group_by_rsu(vehicles, params.M)

    # We'll run MACTER on the largest RSU group for clarity
    all_vs = max(groups.values(), key=len)
    if len(all_vs) < 2:
        # fallback: use all vehicles as one group
        all_vs = vehicles

    rsu_F = params.F_vec_total_GHz

    # Initial allocation
    alloc, decisions, _ = algorithm1_resource_allocation_and_choice(all_vs, rsu_F)
    iters_util = [_avg_utility(all_vs, alloc, decisions)]
    iters      = [0]

    for it in range(1, params.max_iter_algo2 + 1):
        prev = decisions.copy()
        offloaders = [v for v in all_vs if decisions[v.vid] == "vec"]
        alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        changed = False
        for v in all_vs:
            u_loc = utility_local(v)
            if decisions[v.vid] == "vec":
                f_v = alloc.get(v.vid, 0.0)
                u_vec = utility_vec(v, f_v)
            else:
                cand = offloaders + [v]
                ca   = allocate_edge_cpu_bisection(cand, rsu_F, tol=1e-3)
                u_vec = utility_vec(v, ca.get(v.vid, 0.0))

            new_dec = "vec" if u_vec > u_loc else "loc"
            if new_dec != decisions[v.vid]:
                decisions[v.vid] = new_dec
                changed = True
                offloaders = [vv for vv in all_vs if decisions[vv.vid] == "vec"]
                alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        iters_util.append(_avg_utility(all_vs, alloc, decisions))
        iters.append(it)

        if not changed or decisions == prev:
            break

    # Random and all-local baselines for the same vehicle set
    rand_utils = []
    for t in range(30):
        al, dc, _ = random_offload(all_vs, rsu_F, seed=t)
        rand_utils.append(_avg_utility(all_vs, al, dc))

    al_loc, dc_loc, _ = all_local(all_vs, rsu_F)
    local_util = _avg_utility(all_vs, al_loc, dc_loc)

    return dict(
        macter_iters=iters,
        macter_utils=iters_util,
        rand_iters=iters,
        rand_utils=_mean(rand_utils),
        local_util=local_util,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out = "macter_vs_random.png"
    pct_rand, pct_local = build_figure(out)
    print(f"\n{'='*60}")
    print(f"  MACTER vs Random  : +{pct_rand:.1f}% mean computation efficiency")
    print(f"  MACTER vs All-Local: +{pct_local:.1f}% mean computation efficiency")
    print(f"{'='*60}")