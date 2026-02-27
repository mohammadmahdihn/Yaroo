from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Tuple, Optional



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
    while True:
        x = random.gauss(mu, sigma)
        if lo <= x <= hi:
            return x



@dataclass
class SystemParams:
    M: int = 5                    # RSUs
    r: float = 200.0              # RSU comm "radius" (m)
    e: float = 100.0              # vertical distance road-RSU (m)
    mmwave_range: float = 150.0   # (m)
    cellular_range: float = 200.0 # (m)

    # RSU compute
    F_vec_total_GHz: float = 80.0 # total edge CPU per RSU (GHz)

    # Max RSU data rate cap
    rsu_max_data_rate_bps: float = 1e9

    # Cellular communication
    W_uu: float = 20e6
    cellular_path_loss_exp: float = 3.2
    N0_dbm_per_hz: float = -174.0
    NF_cell_db: float = 7.0

    # mmWave communication
    W_mm: float = 200e6
    mmwave_path_loss_exp: float = 3.2
    NF_mm_db: float = 7.0
    shadow_fading_db: float = 3.0
    G_vehicle_db: float = 15.0
    G_rsu_db: float = 15.0
    pl_const_db: float = 69.6

    # Task and compute
    alpha_kB_min: float = 20.0
    alpha_kB_max: float = 60.0
    xi_min: float = 0.2        # Gcycles/kB
    xi_max: float = 0.4        # Gcycles/kB
    tmax_min: float = 0.2      # s
    tmax_max: float = 1.0      # s
    f_loc_min: float = 10.0    # GHz
    f_loc_max: float = 15.0    # GHz

    # Vehicle speed (km/h)
    mu_kmh: float = 150.0
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

    # Segment length along road for each RSU coverage
    @property
    def seg_len(self) -> float:
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
        return self.xi_Gcycles_per_kB * self.alpha_kB


@dataclass
class Vehicle:
    vid: int
    params: SystemParams
    s: float  # position
    v_kmh: float
    f_loc_GHz: float
    zeta_J_per_cycle: float
    task: Task

    rsu_id: int = field(init=False, default=0)
    dist_to_rsu_m: float = field(init=False, default=0.0)
    rsu_coverage_time: float = field(init=False, default=0.0)
    link_type: str = field(init=False, default="cellular")
    rate_bps: float = field(init=False, default=0.0)

    def v_ms(self) -> float:
        return self.v_kmh * 1000.0 / 3600.0

    def stay_time(self) -> float:
        return self.params.seg_len / max(self.v_ms(), 1e-6)

    def practical_tolerable_delay(self) -> float:
        return min(self.task.t_max, self.stay_time())

    def assign_rsu_and_distance(self) -> None:
        seg = int(self.s / self.params.seg_len)
        seg = clamp(seg, 0, self.params.M - 1)
        self.rsu_id = int(seg)

        center_x = self.rsu_id * self.params.seg_len + self.params.seg_len / 2.0
        horiz = abs(self.s - center_x)
        self.dist_to_rsu_m = math.sqrt(horiz * horiz + self.params.e * self.params.e)
        self.rsu_coverage_time = (((self.rsu_id + 1) * self.params.seg_len) - self.s) / self.v_ms()

    def compute_uplink_rate(self) -> None:
        self.assign_rsu_and_distance()

        d = max(self.dist_to_rsu_m, 1.0)

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
        noise_dbm = self.params.N0_dbm_per_hz + self.params.NF_cell_db + 10.0 * math.log10(self.params.W_uu)
        noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)

        h_abs_sq = random.expovariate(1.0)         # Rayleigh fading

        snr = (self.params.p_tx_w * (d ** (-self.params.cellular_path_loss_exp)) * h_abs_sq) / max(noise_w, 1e-18)
        return self.params.W_uu * log2(1.0 + snr)

    def _rate_mmwave(self, d: float) -> float:
        p_dbm = w_to_dbm(self.params.p_tx_w)
        noise_dbm = self.params.N0_dbm_per_hz + self.params.NF_mm_db + 10.0 * math.log10(self.params.W_mm)

        path_loss_db = 10.0 * self.params.mmwave_path_loss_exp * math.log10(d) + self.params.pl_const_db + self.params.shadow_fading_db
        snr_db = (p_dbm - noise_dbm) + (self.params.G_vehicle_db + self.params.G_rsu_db) - path_loss_db

        snr_lin = db_to_linear(snr_db)
        return self.params.W_mm * log2(1.0 + snr_lin)

    def local_time(self) -> float:
        return self.task.C_Gcycles / max(self.f_loc_GHz, 1e-9)

    def local_energy_J(self) -> float:
        return self.zeta_J_per_cycle * (self.task.C_Gcycles * 1e9)

    def vec_time(self, f_vec_GHz: float) -> float:
        if self.rate_bps <= 0.0:
            return float("inf")
        tx = self.task.alpha_bits / self.rate_bps
        exe = self.task.C_Gcycles / max(f_vec_GHz, 1e-9)
        return tx + exe

    def vec_energy_J(self) -> float:
        if self.rate_bps <= 0.0:
            return float("inf")
        return self.params.p_tx_w * (self.task.alpha_bits / self.rate_bps)


def utility_local(v: Vehicle) -> float:
    p = v.params
    K_loc = p.wE * v.local_energy_J() + p.wT * v.local_time()
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

    e_vec_max = p.zeta_max * (v.task.C_Gcycles * 1e9)
    X_vec = p.wE * e_vec_max + p.wT * v.practical_tolerable_delay()

    slack = X_vec - K_vec
    sat = p.theta * math.log1p(max(slack, 0.0))
    cost = (1.0 - p.theta) * p.rho_vec * f_vec_GHz
    return sat - cost


def _f_opt_for_lambda(v: Vehicle, lam: float) -> float:
    p = v.params
    if v.rate_bps <= 0.0:
        return 0.0

    e_vec     = v.vec_energy_J()
    e_vec_max = v.params.zeta_max * (v.task.C_Gcycles * 1e9)
    t_ptd     = v.practical_tolerable_delay()
    tx_time   = v.task.alpha_bits / v.rate_bps

    A = p.wE * (e_vec_max - e_vec) + p.wT * (t_ptd - tx_time)
    C_i = p.wT * v.task.C_Gcycles

    if A <= 0.0 or C_i <= 0.0:
        return 0.0

    denominator = (1.0 - p.theta) * p.rho_vec + lam
    if denominator <= 0.0:
        denominator = 1e-12

    a = 1.0 + A
    inside = C_i * C_i + 4.0 * a * p.theta * C_i / denominator
    f = (C_i + math.sqrt(max(inside, 0.0))) / (2.0 * a)

    if A - C_i / max(f, 1e-12) <= 0.0:
        return 0.0

    return max(f, 0.0)

def allocate_edge_cpu_bisection(vehicles_offload: List[Vehicle], F_total_GHz: float, tol: float) -> Dict[int, float]:
    if not vehicles_offload:
        return {}

    lam_low = 0.0
    lam_high = 1.0
    for _ in range(60):
        s = sum(_f_opt_for_lambda(v, lam_high) for v in vehicles_offload)
        if s <= F_total_GHz:
            break
        lam_high *= 2.0
        if lam_high > 1e12:
            break

    for _ in range(80):
        lam_mid = 0.5 * (lam_low + lam_high)
        s = sum(_f_opt_for_lambda(v, lam_mid) for v in vehicles_offload)

        if abs(s - F_total_GHz) <= tol:
            break

        if s > F_total_GHz:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    lam_star = lam_high
    alloc = {v.vid: _f_opt_for_lambda(v, lam_star) for v in vehicles_offload}

    return alloc



def algorithm1_resource_allocation_and_choice(vehicles: List[Vehicle], rsu_F: float) -> Tuple[Dict[int, float], Dict[int, str], Dict[int, Tuple[float, float]]]:
    alloc_all = allocate_edge_cpu_bisection(vehicles, rsu_F, tol=1e-3)

    decisions: Dict[int, str] = {}
    utilities: Dict[int, Tuple[float, float]] = {}
    chosen_alloc: Dict[int, float] = {}

    for v in vehicles:
        u_loc = utility_local(v)
        f_v = alloc_all.get(v.vid, 0.0)
        u_vec = utility_vec(v, f_v)

        utilities[v.vid] = (u_loc, u_vec)

        if u_vec == float("-inf") and u_loc == float("-inf"):
            decisions[v.vid] = "loc"
            continue

        if u_vec > u_loc:
            decisions[v.vid] = "vec"
            chosen_alloc[v.vid] = f_v
        else:
            decisions[v.vid] = "loc"

    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    final_alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

    for v in offloaders:
        u_loc, _ = utilities[v.vid]
        utilities[v.vid] = (u_loc, utility_vec(v, final_alloc.get(v.vid, 0.0)))

    return final_alloc, decisions, utilities


def distributed_macter(vehicles: List[Vehicle], rsu_F: float, max_iter: int) -> Tuple[Dict[int, float], Dict[int, str], float]:
    alloc, decisions, _ = algorithm1_resource_allocation_and_choice(vehicles, rsu_F)

    for _ in range(max_iter):
        prev = decisions.copy()

        offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
        alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        changed = False
        for v in vehicles:
            u_loc = utility_local(v)
            if decisions[v.vid] == "vec":
                f_v = alloc.get(v.vid, 0.0)
                u_vec = utility_vec(v, f_v)
            else:
                candidate_off = offloaders + [v]
                candidate_alloc = allocate_edge_cpu_bisection(candidate_off, rsu_F, tol=1e-3)
                u_vec = utility_vec(v, candidate_alloc.get(v.vid, 0.0))

            new_dec = "vec" if (u_vec > u_loc) else "loc"
            if new_dec != decisions[v.vid]:
                decisions[v.vid] = new_dec
                changed = True

                offloaders = [vv for vv in vehicles if decisions[vv.vid] == "vec"]
                alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

        if not changed or decisions == prev:
            break

    offloaders = [v for v in vehicles if decisions[v.vid] == "vec"]
    alloc = allocate_edge_cpu_bisection(offloaders, rsu_F, tol=1e-3)

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
            if v.rsu_id >= v.params.M - 1:
                groups[v.rsu_id].append(v)
                continue
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
    result = transmission_time + minimum_comp_time > v.rsu_coverage_time and v.rsu_coverage_time + v.local_time() < v.task.t_max
    return result