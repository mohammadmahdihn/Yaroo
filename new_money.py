"""
MACTER (Mobility-Aware Computational Efficiency-Based Task Offloading and Resource Allocation)
Based on: "Task Offloading and Resource Allocation for IoV Using 5G NR-V2X Communication" (IEEE IoTJ 2022)

This file implements:
- Mobility model (truncated Gaussian speed, stay time)
- 5G NR-V2X V2I comm model: cellular + mmWave (rate depends on distance)
- Computation model: local vs VEC (time + energy)
- Utility functions (paper Eq. 15,16 style)
- Algorithm 1: resource allocation (KKT + bisection on lambda) + offloading choice
- Algorithm 2: distributed MACTER (iterative strategy updates)

No SciPy required.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Tuple, Optional


# ----------------------------
# Helpers
# ----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def w_to_dbm(p_w: float) -> float:
    # watts -> dBm
    return 10.0 * math.log10(max(p_w, 1e-18) * 1000.0)

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)

def log2(x: float) -> float:
    return math.log(x, 2)

def trunc_gauss(mu: float, sigma: float, lo: float, hi: float) -> float:
    # rejection sampling
    while True:
        x = random.gauss(mu, sigma)
        if lo <= x <= hi:
            return x


# ----------------------------
# Parameters
# ----------------------------

@dataclass
class SystemParams:
    # Topology
    M: int = 5                    # RSUs
    r: float = 200.0              # RSU comm "radius" (m)
    e: float = 100.0              # vertical distance road-RSU (m)
    mmwave_range: float = 150.0   # (m)
    cellular_range: float = 200.0 # (m)

    # RSU compute
    F_vec_total_GHz: float = 80.0 # total edge CPU per RSU (GHz)

    # Max RSU data rate cap
    rsu_max_data_rate_bps: float = 1e9

    # Cellular comm
    W_uu: float = 20e6                      # Hz
    cellular_path_loss_exp: float = 3.2
    N0_dbm_per_hz: float = -174.0
    NF_cell_db: float = 7.0

    # mmWave comm
    W_mm: float = 200e6                     # Hz
    mmwave_path_loss_exp: float = 3.2
    NF_mm_db: float = 7.0
    shadow_fading_db: float = 3.0
    G_vehicle_db: float = 15.0
    G_rsu_db: float = 15.0
    pl_const_db: float = 69.6               # from paper Eq(7)

    # Task and compute
    alpha_kB_min: float = 20.0
    alpha_kB_max: float = 60.0
    xi_min: float = 0.2        # Gcycles/kB
    xi_max: float = 0.4        # Gcycles/kB
    tmax_min: float = 0.2      # s
    tmax_max: float = 1.0      # s
    f_loc_min: float = 10.0    # GHz
    f_loc_max: float = 15.0    # GHz

    # Vehicle mobility (km/h)
    mu_kmh: float = 60.0
    sigma_kmh: float = 10.0

    # Energy model
    zeta_min: float = 1e-9     # J per cycle
    zeta_max: float = 3e-9

    # Offloading
    p_tx_w: float = 1.3        # W
    theta: float = 0.5         # weight coefficient in U_vec
    rho_vec: float = 0.01      # cost per GHz (unit cost)
    wE: float = 0.5            # weight for energy in ETC
    wT: float = 0.5            # weight for time in ETC

    # Utility penalty for infeasible local
    eta_penalty: float = 10.0

    # Numerical
    lambda_tol: float = 1e-4
    max_lambda: float = 1e6
    max_iter_algo2: int = 50

    # A pragmatic "budget" factor when paper doesn't specify hard max energy for offloading
    # e_vec_max = offload_energy_budget_factor * e_vec (makes constraint meaningful)
    offload_energy_budget_factor: float = 2.0

    # Segment length along road for each RSU coverage
    @property
    def seg_len(self) -> float:
        # horizontal chord length for coverage given radius r and vertical offset e
        return 2.0 * math.sqrt(max(self.r * self.r - self.e * self.e, 1e-9))

    @property
    def road_len(self) -> float:
        return self.M * self.seg_len

    @property
    def vmin_kmh(self) -> float:
        return self.mu_kmh - 3.0 * self.sigma_kmh

    @property
    def vmax_kmh(self) -> float:
        return self.mu_kmh + 3.0 * self.sigma_kmh


# ----------------------------
# Models
# ----------------------------

@dataclass
class Task:
    alpha_kB: float
    xi_Gcycles_per_kB: float
    t_max: float

    @property
    def alpha_bits(self) -> float:
        return self.alpha_kB * 1024.0 * 8.0

    @property
    def C_Gcycles(self) -> float:
        # Ci = xi * alpha_in (paper), with units consistent with GHz CPU
        return self.xi_Gcycles_per_kB * self.alpha_kB


@dataclass
class Vehicle:
    vid: int
    params: SystemParams
    s: float  # position along road (m)
    v_kmh: float
    f_loc_GHz: float
    zeta_J_per_cycle: float
    task: Task

    # computed each snapshot
    rsu_id: int = field(init=False, default=0)
    dist_to_rsu_m: float = field(init=False, default=0.0)
    rsu_coverage_time: float = field(init=False, default=0.0)
    link_type: str = field(init=False, default="cellular")
    rate_bps: float = field(init=False, default=0.0)

    def v_ms(self) -> float:
        return self.v_kmh * 1000.0 / 3600.0

    def stay_time(self) -> float:
        # paper Eq(3): 2*sqrt(r^2 - e^2) / v
        return self.params.seg_len / max(self.v_ms(), 1e-6)

    def practical_tolerable_delay(self) -> float:
        return min(self.task.t_max, self.stay_time())

    def assign_rsu_and_distance(self) -> None:
        # which RSU segment
        seg = int(self.s / self.params.seg_len)
        seg = clamp(seg, 0, self.params.M - 1)
        self.rsu_id = int(seg)

        # center of that RSU segment along the road
        center_x = self.rsu_id * self.params.seg_len + self.params.seg_len / 2.0
        horiz = abs(self.s - center_x)
        self.dist_to_rsu_m = math.sqrt(horiz * horiz + self.params.e * self.params.e)
        self.rsu_coverage_time = (((self.rsu_id + 1) * self.params.seg_len) - self.s) / self.v_ms()

    def compute_uplink_rate(self) -> None:
        self.assign_rsu_and_distance()

        d = max(self.dist_to_rsu_m, 1.0)

        # choose link based on distance: mmWave preferred within its range
        if d <= self.params.mmwave_range:
            self.link_type = "mmwave"
            rate = self._rate_mmwave(d)
        elif d <= self.params.cellular_range:
            self.link_type = "cellular"
            rate = self._rate_cellular(d)
        else:
            self.link_type = "out"
            rate = 0.0

        self.rate_bps = min(rate, self.params.rsu_max_data_rate_bps)

    def _rate_cellular(self, d: float) -> float:
        # paper Eq(4) style: W log2(1 + SNR)
        # noise power: N0 + NF + 10log10(W)
        noise_dbm = self.params.N0_dbm_per_hz + self.params.NF_cell_db + 10.0 * math.log10(self.params.W_uu)
        noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        # Rayleigh fading |h|^2 ~ exp(1)
        h_abs_sq = random.expovariate(1.0)

        snr = (self.params.p_tx_w * (d ** (-self.params.cellular_path_loss_exp)) * h_abs_sq) / max(noise_w, 1e-18)
        return self.params.W_uu * log2(1.0 + snr)

    def _rate_mmwave(self, d: float) -> float:
        # paper Eq(6)(7) style in dB
        p_dbm = w_to_dbm(self.params.p_tx_w)
        noise_dbm = self.params.N0_dbm_per_hz + self.params.NF_mm_db + 10.0 * math.log10(self.params.W_mm)

        path_loss_db = 10.0 * self.params.mmwave_path_loss_exp * math.log10(d) + self.params.pl_const_db + self.params.shadow_fading_db
        snr_db = (p_dbm - noise_dbm) + (self.params.G_vehicle_db + self.params.G_rsu_db) - path_loss_db

        snr_lin = db_to_linear(snr_db)
        return self.params.W_mm * log2(1.0 + snr_lin)

    # -------- local compute --------
    def local_time(self) -> float:
        return self.task.C_Gcycles / max(self.f_loc_GHz, 1e-9)

    def local_energy_J(self) -> float:
        # e_loc = zeta * cycles, cycles = C_Gcycles * 1e9
        return self.zeta_J_per_cycle * (self.task.C_Gcycles * 1e9)

    # -------- edge compute --------
    def vec_time(self, f_vec_GHz: float) -> float:
        if self.rate_bps <= 0.0:
            return float("inf")
        tx = self.task.alpha_bits / self.rate_bps
        exe = self.task.C_Gcycles / max(f_vec_GHz, 1e-9)
        return tx + exe

    def vec_energy_J(self) -> float:
        if self.rate_bps <= 0.0:
            return float("inf")
        # paper Eq(11): p * alpha / gamma
        return self.params.p_tx_w * (self.task.alpha_bits / self.rate_bps)


# ----------------------------
# Utility functions (paper-like)
# ----------------------------

def utility_local(v: Vehicle) -> float:
    p = v.params
    K_loc = p.wE * v.local_energy_J() + p.wT * v.local_time()
    # max energy for local: use worst zeta (paper hint)
    e_loc_max = p.zeta_max * (v.task.C_Gcycles * 1e9)
    X_loc = p.wE * e_loc_max + p.wT * v.task.t_max

    slack = X_loc - K_loc
    sat = math.log1p(max(slack, 0.0))
    penalty = p.eta_penalty if (X_loc < K_loc) else 0.0
    return sat - penalty

def utility_vec(v: Vehicle, f_vec_GHz: float) -> float:
    p = v.params
    if v.rate_bps <= 0.0 or f_vec_GHz <= 0.0:
        return float("-inf")

    K_vec = p.wE * v.vec_energy_J() + p.wT * v.vec_time(f_vec_GHz)

    # FIX: e_vec_max is the vehicle's MAX ENERGY BUDGET for this task, i.e., the
    # maximum energy the vehicle is allowed to spend while offloading.
    # The paper states it is "highly dependent on transmission rate", but its role
    # in the utility (as an upper bound in the slack) is analogous to e_loc_max:
    # it is the worst-case / maximum-allowed energy expenditure for the task.
    # Using zeta_max * C puts it on the same scale as e_loc_max, making
    # utility_vec and utility_local comparable -- which is required for a meaningful
    # offloading decision.
    #
    # BUG THAT WAS HERE:
    #   e_vec_max = budget_factor * vec_energy_J()          <- ~2 * e_vec (milliJoules)
    #               + alpha_bits/rate_bps * p_tx_w          <- + e_vec AGAIN (double-count)
    # This made e_vec_max ≈ 3 * e_vec ≈ 12 mJ, which is ~3000x smaller than
    # e_loc_max (~36 J).  Consequently X_vec ≈ 0.26 vs X_loc ≈ 18.25, making
    # the VEC slack (≈0.013) negligible and utility_vec always << utility_local.
    e_vec_max = p.zeta_max * (v.task.C_Gcycles * 1e9)   # same scale as e_loc_max
    X_vec = p.wE * e_vec_max + p.wT * v.practical_tolerable_delay()

    slack = X_vec - K_vec
    sat = p.theta * math.log1p(max(slack, 0.0))
    cost = (1.0 - p.theta) * p.rho_vec * f_vec_GHz
    return sat - cost


# ----------------------------
# Algorithm 1: Resource allocation (KKT + bisection)
# ----------------------------

def _f_opt_for_lambda(v: Vehicle, lam: float) -> float:
    """
    Closed-form from KKT for our paper-consistent objective.
    Derived using:
      U_vec = theta*ln(1 + (X - K)^+) - (1-theta)*rho*f
      K depends on f via C/f term.
    """
    p = v.params
    if v.rate_bps <= 0.0:
        return 0.0

    # FIX: use the same corrected e_vec_max as in utility_vec.
    # OLD (buggy): e_vec_max = budget_factor*e_vec + e_vec  (double-count, milliJoule scale)
    # This made A tiny (~0.25), so C_i/A was large (~24 GHz), and the KKT f
    # often barely satisfied or failed the slack check, forcing the return of 0.
    # With the fix, A is dominated by wE*e_loc_max (~18), so C_i/A ~0.33 GHz and
    # virtually every reasonable allocation yields positive slack.
    e_vec     = v.vec_energy_J()
    e_vec_max = v.params.zeta_max * (v.task.C_Gcycles * 1e9)   # same scale as e_loc_max
    t_ptd     = v.practical_tolerable_delay()
    tx_time   = v.task.alpha_bits / v.rate_bps

    A = p.wE * (e_vec_max - e_vec) + p.wT * (t_ptd - tx_time)
    C_i = p.wT * v.task.C_Gcycles

    # If A <= 0, even infinite CPU can't make slack positive -> no point allocating
    if A <= 0.0 or C_i <= 0.0:
        return 0.0

    # KKT leads to:
    # f = ( C_i + sqrt(C_i^2 + 4*(1+A)*theta*C_i / ((1-theta)*rho + lam)) ) / (2*(1+A))
    denom = (1.0 - p.theta) * p.rho_vec + lam
    if denom <= 0.0:
        denom = 1e-12

    a = 1.0 + A
    inside = C_i * C_i + 4.0 * a * p.theta * C_i / denom
    f = (C_i + math.sqrt(max(inside, 0.0))) / (2.0 * a)

    # Ensure f yields positive slack (otherwise utility is just cost, so clamp to 0)
    if A - C_i / max(f, 1e-12) <= 0.0:
        return 0.0

    return max(f, 0.0)

def allocate_edge_cpu_bisection(vehicles_offload: List[Vehicle], F_total_GHz: float, tol: float) -> Dict[int, float]:
    """
    Bisection on lambda so sum f_i <= F_total.
    """
    if not vehicles_offload:
        return {}

    # Find lambda_high such that sum f(lambda_high) <= F_total
    lam_low = 0.0
    lam_high = 1.0
    for _ in range(60):
        s = sum(_f_opt_for_lambda(v, lam_high) for v in vehicles_offload)
        if s <= F_total_GHz:
            break
        lam_high *= 2.0
        if lam_high > 1e12:
            break

    # Bisection
    for _ in range(80):
        lam_mid = 0.5 * (lam_low + lam_high)
        s = sum(_f_opt_for_lambda(v, lam_mid) for v in vehicles_offload)

        if abs(s - F_total_GHz) <= tol:
            lam_low = lam_mid
            break

        if s > F_total_GHz:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = lam_high  # FIX: lam_low gives sum slightly > F_total (over-allocation).
    # lam_high is the smallest lambda where sum <= F_total,
    # which properly satisfies the resource constraint C3.
    alloc = {v.vid: _f_opt_for_lambda(v, lam_star) for v in vehicles_offload}

    # If numerical slack leaves unused CPU, that's fine. (paper uses <= constraint)
    return alloc


# ----------------------------
# Algorithm 1 wrapper: allocation + per-vehicle offload choice
# ----------------------------

def algorithm1_resource_allocation_and_choice(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], Dict[int, Tuple[float, float]]]:
    """
    Runs resource allocation assuming current offload set is "all vehicles try VEC".
    Then computes U_loc and U_vec and picks best per vehicle.

    Returns:
      f_alloc (only for those choosing vec),
      decisions: vid -> "loc"/"vec"
      utilities: vid -> (U_loc, U_vec_used)
    """
    # Allocate edge CPU as if everyone offloads (Algorithm 1's first stage vibe)
    alloc_all = allocate_edge_cpu_bisection(vehicles, rsu_F, tol=1e-3)

    decisions: Dict[int, str] = {}
    utilities: Dict[int, Tuple[float, float]] = {}
    chosen_alloc: Dict[int, float] = {}

    for v in vehicles:
        u_loc = utility_local(v)
        f_v = alloc_all.get(v.vid, 0.0)
        u_vec = utility_vec(v, f_v)

        utilities[v.vid] = (u_loc, u_vec)

        # paper’s feasibility check spirit:
        # if both are awful (infeasible) -> choose none (we map to local to keep simulation moving)
        if u_vec == float("-inf") and u_loc == float("-inf"):
            decisions[v.vid] = "loc"
            continue

        if u_vec > u_loc:
            decisions[v.vid] = "vec"
            chosen_alloc[v.vid] = f_v
        else:
            decisions[v.vid] = "loc"

    # Re-allocate CPU among the actually-offloading vehicles (more realistic and matches Algo2 iterations)
    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    final_alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

    # Update U_vec with the final allocation for reporting
    for v in offloaders:
        u_loc, _ = utilities[v.vid]
        utilities[v.vid] = (u_loc, utility_vec(v, final_alloc.get(v.vid, 0.0)))

    return final_alloc, decisions, utilities


# ----------------------------
# Algorithm 2: Distributed MACTER
# ----------------------------

def distributed_macter(vehicles: List[Vehicle], rsu_F: float, max_iter: int) -> Tuple[Dict[int, float], Dict[int, str], float]:
    """
    Distributed MACTER style iterative process.
    We do best-response updates (practical, converges well).
    """
    # Initial: run Algo1 choice once
    alloc, decisions, _ = algorithm1_resource_allocation_and_choice(vehicles, rsu_F)

    for _ in range(max_iter):
        prev = decisions.copy()

        # Allocate based on current offload set
        offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
        alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        # Sequential best response updates (distributed-ish)
        changed = False
        for v in vehicles:
            u_loc = utility_local(v)

            # utility if VEC (if we switch v to vec, re-allocate on that RSU set)
            if decisions[v.vid] == "vec":
                f_v = alloc.get(v.vid, 0.0)
                u_vec = utility_vec(v, f_v)
            else:
                # candidate set = current offloaders + v
                cand_off = offloaders + [v]
                cand_alloc = allocate_edge_cpu_bisection(cand_off, rsu_F, tol=1e-3)
                u_vec = utility_vec(v, cand_alloc.get(v.vid, 0.0))

            new_dec = "vec" if (u_vec > u_loc) else "loc"
            if new_dec != decisions[v.vid]:
                decisions[v.vid] = new_dec
                changed = True

                # refresh offloaders list for subsequent players (best-response dynamics)
                offloaders = [vv for vv in vehicles if decisions[vv.vid] == "vec"]
                alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        if not changed or decisions == prev:
            break

    # Final allocation with final decisions
    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

    # Compute system computation efficiency (paper Eq 17-ish; we use total chosen utility per total energy)
    total_utility = 0.0
    total_energy = 1e-12

    for v in vehicles:
        if decisions[v.vid] == "vec":
            f_v = alloc.get(v.vid, 0.0)
            u = utility_vec(v, f_v)
            e = v.local_energy_J() + v.vec_energy_J()
        else:
            u = utility_local(v)
            e = v.local_energy_J()

        total_utility += u
        total_energy += max(e, 1e-12)

    E = total_utility / total_energy
    return alloc, decisions, E


# ----------------------------
# Simulation setup
# ----------------------------

def make_vehicles(N: int, params: SystemParams, seed: Optional[int] = 7) -> List[Vehicle]:
    if seed is not None:
        random.seed(seed)

    vehicles: List[Vehicle] = []
    for i in range(N):
        s = random.uniform(0.0, params.road_len)
        v_kmh = trunc_gauss(params.mu_kmh, params.sigma_kmh, params.vmin_kmh, params.vmax_kmh)
        f_loc = random.uniform(params.f_loc_min, params.f_loc_max)
        zeta = random.uniform(params.zeta_min, params.zeta_max)

        alpha_kB = random.uniform(params.alpha_kB_min, params.alpha_kB_max)
        xi = random.uniform(params.xi_min, params.xi_max)  # Gcycles/kB
        t_max = random.uniform(params.tmax_min, params.tmax_max)

        task = Task(alpha_kB=alpha_kB, xi_Gcycles_per_kB=xi, t_max=t_max)
        v = Vehicle(vid=i, params=params, s=s, v_kmh=v_kmh, f_loc_GHz=f_loc, zeta_J_per_cycle=zeta, task=task)
        v.compute_uplink_rate()
        vehicles.append(v)

    return vehicles


def group_by_rsu(vehicles: List[Vehicle], M: int) -> Dict[int, List[Vehicle]]:
    groups: Dict[int, List[Vehicle]] = {m: [] for m in range(M)}
    for v in vehicles:
        groups[v.rsu_id].append(v)
    return groups

def group_by_rsu_extended(vehicles: List[Vehicle], M: int) -> Dict[int, List[Vehicle]]:
    groups = {m: [] for m in range(M)}

    for v in vehicles:
        if can_not_offload_to_this_rsu(v):
            v2 = copy.deepcopy(v)
            v2.rsu_id += 1
            v2.s = v2.rsu_id * v.params.seg_len + 1e-3
            v2.assign_rsu_and_distance()
            v2.compute_uplink_rate()
            v2.task.t_max = v.task.t_max - v.rsu_coverage_time

            groups[v2.rsu_id].append(v2)
        else:
            groups[v.rsu_id].append(v)

    return groups

def can_not_offload_to_this_rsu(v):
    transmission_time = v.task.alpha_bits / v.rate_bps
    minimum_comp_time = v.task.C_Gcycles / v.params.F_vec_total_GHz
    result = transmission_time + minimum_comp_time > v.rsu_coverage_time > v.local_time() and v.rsu_coverage_time < v.task.t_max
    return result
# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    params = SystemParams()
    N = 30

    vehicles = make_vehicles(N, params, seed=None)
    groups = group_by_rsu(vehicles, params.M)

    all_alloc: Dict[int, float] = {}
    all_dec: Dict[int, str] = {}
    total_E_parts: List[float] = []

    for rsu_id, vs in groups.items():
        if not vs:
            continue

        alloc, dec, E = distributed_macter(vs, rsu_F=params.F_vec_total_GHz, max_iter=params.max_iter_algo2)
        total_E_parts.append(E)

        all_alloc.update(alloc)
        all_dec.update(dec)

        print(f"\nRSU {rsu_id}: vehicles={len(vs)} | efficiency={E:.6e}")
        for v in vs:
            f = all_alloc.get(v.vid, 0.0)
            print(
                f"  v{v.vid:02d} pos={v.s:7.1f}m v={v.v_kmh:5.1f}km/h "
                f"link={v.link_type:8s} d={v.dist_to_rsu_m:6.1f}m rate={v.rate_bps/1e6:7.1f}Mbps "
                f"t_ptd={v.practical_tolerable_delay():.3f}s decision={dec[v.vid]:3s} f_vec={f:6.2f}GHz"
            )

    # crude overall indicator
    if total_E_parts:
        print(f"\nOverall mean efficiency across RSUs: {sum(total_E_parts)/len(total_E_parts):.6e}")