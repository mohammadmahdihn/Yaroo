from __future__ import annotations

import copy
import math
import random
import sys
import os
import statistics
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
from new_money import (
    SystemParams, Vehicle, Task,
    utility_local, utility_vec,
    allocate_edge_cpu_bisection,
    distributed_macter,
    make_vehicles, group_by_rsu, group_by_rsu_extended,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def random_offload(vehicles: List[Vehicle], rsu_F: float,
                   seed: Optional[int] = None) -> Tuple[Dict[int, float], Dict[int, str], float]:
    rng = random.Random(seed)
    decisions: Dict[int, str] = {}

    for v in vehicles:
        if v.rate_bps <= 0.0:
            decisions[v.vid] = "loc"
        else:
            decisions[v.vid] = "vec" if rng.random() < 0.5 else "loc"

    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    n_off = len(offloaders)

    if n_off > 0:
        f_each = rsu_F / n_off
        for v in offloaders:
            if v.vec_time(f_each) > v.practical_tolerable_delay():
                decisions[v.vid] = "loc"

        offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
        n_off = len(offloaders)

    alloc: Dict[int, float] = {}
    if n_off > 0:
        f_each = rsu_F / n_off
        for v in offloaders:
            alloc[v.vid] = f_each

    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)


def all_local(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], float]:
    decisions = {v.vid: "loc" for v in vehicles}
    alloc: Dict[int, float] = {}
    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)


def all_vec(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], float]:
    eligible = [v for v in vehicles if v.rate_bps > 0.0]
    ineligible = [v for v in vehicles if v.rate_bps <= 0.0]

    decisions: Dict[int, str] = {}
    for v in ineligible:
        decisions[v.vid] = "loc"

    n = len(eligible)
    if n > 0:
        f_each = rsu_F / n
        for v in eligible:
            if v.vec_time(f_each) <= v.practical_tolerable_delay():
                decisions[v.vid] = "vec"
            else:
                decisions[v.vid] = "loc"

        feasible = [v for v in eligible if decisions[v.vid] == "vec"]
        n_f = len(feasible)
        if n_f > 0:
            f_each = rsu_F / n_f
            for v in feasible:
                if v.vec_time(f_each) > v.practical_tolerable_delay():
                    decisions[v.vid] = "loc"

    alloc: Dict[int, float] = {}
    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    if offloaders:
        f_each = rsu_F / len(offloaders)
        for v in offloaders:
            alloc[v.vid] = f_each

    return alloc, decisions, _compute_efficiency(vehicles, alloc, decisions)



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


TRIALS = 10


def _run_one_scenario(vehicles: List[Vehicle], params: SystemParams, run_seed: int = 0) -> dict:

    groups = group_by_rsu(vehicles, params.M)
    vehicles_copy = copy.deepcopy(vehicles)
    groups_ext = group_by_rsu_extended(vehicles_copy, params.M)

    # ── MACTER-EXTENDED ────────────────────────────────────────────────────────────────
    m_ex_eff, m_ex_util, m_ex_vratio, m_ex_delay = [], [], [], []
    for vs in groups_ext.values():
        if not vs:
            continue
        alloc, dec, eff = distributed_macter(
            vs, params.F_vec_total_GHz, params.max_iter_algo2
        )
        m_ex_eff.append(eff)
        m_ex_util.append(_avg_utility(vs, alloc, dec))
        m_ex_vratio.append(_vec_ratio(dec))
        m_ex_delay.append(_avg_delay(vs, alloc, dec))

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

    # ── Random ─────────────────────────
    r_eff_t, r_util_t, r_vratio_t, r_delay_t = [], [], [], []
    for trial in range(TRIALS):
        t_eff, t_util, t_vr, t_delay = [], [], [], []
        for vs in groups.values():
            if not vs:
                continue
            alloc, dec, eff = random_offload(vs, params.F_vec_total_GHz, seed=(run_seed + trial) % (2**31))
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
        macter_ex_eff=_mean(m_ex_eff),       macter_ex_util=_mean(m_ex_util),
        macter_ex_vr=_mean(m_ex_vratio),     macter_ex_delay=_mean(m_ex_delay),  # Add extended MACTER

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


def sweep_num_vehicles(N_vals, params: SystemParams, outer_trials: int = 8, run_seed: int = 0):
    results = {k: [] for k in [
        "macter_eff","macter_ex_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
        "macter_util","macter_ex_util","rand_util","rand_util_std","local_util","allvec_util",
        "macter_vr","macter_ex_vr","rand_vr","macter_delay","rand_delay","local_delay","allvec_delay",
    ]}
    for N in N_vals:
        print(f"  sweep_N: N={N}", flush=True)
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, params, seed=(run_seed + t * 1000 + N) % (2**31))
            trial_data.append(_run_one_scenario(vehicles, params, run_seed=run_seed))

        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results



def sweep_data_size(alpha_vals_kB, params: SystemParams, N: int = 20, outer_trials: int = 8, run_seed: int = 0):
    results = {k: [] for k in [
        "macter_ex_eff","macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
    ]}
    for a in alpha_vals_kB:
        print(f"  sweep_alpha: alpha={a}kB", flush=True)
        p = copy.copy(params)
        p.alpha_kB_min = a * 0.85
        p.alpha_kB_max = a * 1.15
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, p, seed=(run_seed + t * 500 + int(a)) % (2**31))
            trial_data.append(_run_one_scenario(vehicles, p, run_seed=run_seed))
        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


def sweep_f_total(f_vals_GHz, params: SystemParams, N: int = 30, outer_trials: int = 8, run_seed: int = 0):
    """Sweep edge CPU budget — shows where the t_ptd feasibility check starts to bite."""
    results = {k: [] for k in [
        "macter_ex_eff","macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
        "macter_vr","rand_vr",
    ]}
    for f in f_vals_GHz:
        print(f"  sweep_F: F={f} GHz", flush=True)
        p = copy.copy(params)
        p.F_vec_total_GHz = f
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, p, seed=(run_seed + t * 700 + int(f)) % (2**31))
            trial_data.append(_run_one_scenario(vehicles, p, run_seed=run_seed))
        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


def sweep_tmax(tmax_vals, params: SystemParams, N: int = 20, outer_trials: int = 8, run_seed: int = 0):
    results = {k: [] for k in [
        "macter_ex_eff", "macter_eff","rand_eff","rand_eff_std","local_eff","allvec_eff",
        "macter_ex_util", "macter_util","rand_util","rand_util_std","local_util","allvec_util",
    ]}
    for tmax in tmax_vals:
        print(f"  sweep_tmax: tmax={tmax:.1f}s", flush=True)
        p = copy.copy(params)
        p.tmax_min = tmax * 0.7
        p.tmax_max = tmax * 1.3
        trial_data = []
        for t in range(outer_trials):
            vehicles = make_vehicles(N, p, seed=(run_seed + t * 300 + int(tmax * 100)) % (2**31))
            trial_data.append(_run_one_scenario(vehicles, p, run_seed=run_seed))
        for k in results:
            base = k.replace("_std", "")
            if k.endswith("_std"):
                results[k].append(_std([d[base] for d in trial_data]))
            else:
                results[k].append(_mean([d[k] for d in trial_data]))
    return results


def per_vehicle_snapshot(vehicles: List[Vehicle], params: SystemParams, run_seed: int = 0):
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
            alloc_r, dec_r, _ = random_offload(vs, params.F_vec_total_GHz, seed=(run_seed + trial) % (2**31))
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



COLORS = {
    "macter":  "#1a6faf",
    "macter_ex":  "#049fa3",
    "random":  "#e05c2a",
    "local":   "#4caa4c",
    "allvec":  "#9b59b6",
}

MARKERS = {"macter": "o", "random": "s", "local": "^", "allvec": "D", "macter_ex": "4"}
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



def build_figure(out_path: str, run_seed: int = 0):
    params = SystemParams()

    # ── Experiment 1: vary N ──────────────────────────────────────────────────
    N_vals = [5, 10, 15, 20, 25, 30, 40, 50]
    print("Running sweep: num_vehicles …")
    res_N = sweep_num_vehicles(N_vals, params, outer_trials=8, run_seed=run_seed)

    # ── Experiment 2: vary data size ──────────────────────────────────────────
    alpha_vals = [10, 20, 30, 40, 50, 60, 70, 80]
    print("Running sweep: data size …")
    res_alpha = sweep_data_size(alpha_vals, params, N=20, outer_trials=8, run_seed=run_seed)

    # ── Experiment 3: vary tmax ───────────────────────────────────────────────
    tmax_vals = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    print("Running sweep: max tolerable delay …")
    res_tmax = sweep_tmax(tmax_vals, params, N=20, outer_trials=8, run_seed=run_seed)

    # ── Experiment 4: per-vehicle snapshot ───────────────────────────────────
    print("Running per-vehicle snapshot …")
    snap_seed = (run_seed + 99991) % (2**31)
    vehicles_snap = make_vehicles(30, params, seed=snap_seed)
    macter_u, rand_u, macter_dec, rand_dec = per_vehicle_snapshot(vehicles_snap, params, run_seed=run_seed)
    vids = sorted(macter_u.keys())
    snap_m = [macter_u[i] for i in vids]
    snap_r = [rand_u[i]   for i in vids]

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 25))
    fig.patch.set_facecolor("#f7f9fc")

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.52, wspace=0.35,
                           top=0.93, bottom=0.05,
                           left=0.07, right=0.97)

    ax = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for a in ax:
        a.set_facecolor("#ffffff")
        a.grid(True, linestyle="--", linewidth=0.55, alpha=0.55, color="#cccccc")
        a.spines[["top", "right"]].set_visible(False)

    # ── Panel 0: Computation Efficiency vs N ─────────────────────────────────
    _plot_line(ax[0], N_vals, res_N["macter_eff"],  res_N["rand_eff_std"],
               "macter", "MACTER")
    _plot_line(ax[0], N_vals, res_N["macter_ex_eff"], None,
               "macter_ex", "MACTER + Extended", linestyle="-.")  # Add extended version
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
    _plot_line(ax[1], N_vals, res_N["macter_ex_util"], None,
               "macter_ex", "MACTER + Extended", linestyle="-.")  # Add extended version
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
    _plot_line(ax[2], alpha_vals, res_alpha["macter_ex_eff"], None,
               "macter_ex", "MACTER + Extended", linestyle="-.")  # Add extended version
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
    _plot_line(ax[3], tmax_vals, res_tmax["macter_ex_eff"], None,
               "macter_ex", "MACTER + Extended", linestyle="-.")  # Add extended version
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



def _convergence_trace(params: SystemParams, N: int = 25, seed: int = 7, run_seed: int = 0) -> dict:
    from new_money import (
        algorithm1_resource_allocation_and_choice,
        allocate_edge_cpu_bisection,
    )

    vehicles = make_vehicles(N, params, seed=(run_seed + seed) % (2**31))
    groups   = group_by_rsu(vehicles, params.M)

    all_vs = max(groups.values(), key=len)
    if len(all_vs) < 2:
        all_vs = vehicles

    rsu_F = params.F_vec_total_GHz

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

    rand_utils = []
    for t in range(30):
        al, dc, _ = random_offload(all_vs, rsu_F, seed=(run_seed + t) % (2**31))
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



if __name__ == "__main__":
    run_seed = int.from_bytes(os.urandom(4), "big")
    print(f"Run seed: {run_seed}  (re-run with RUN_SEED={run_seed} to reproduce)")

    env_seed = os.environ.get("RUN_SEED")
    if env_seed is not None:
        run_seed = int(env_seed)
        print(f"  (overridden by RUN_SEED env var → {run_seed})")

    out = "macter_vs_random.png"
    pct_rand, pct_local = build_figure(out, run_seed=run_seed)
    print(f"\n{'='*60}")
    print(f"  MACTER vs Random  : +{pct_rand:.1f}% mean computation efficiency")
    print(f"  MACTER vs All-Local: +{pct_local:.1f}% mean computation efficiency")
    print(f"{'='*60}")