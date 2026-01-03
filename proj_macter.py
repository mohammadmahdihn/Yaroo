"""
Implementation aligned with:
Salman Raza et al., "Task Offloading and Resource Allocation for IoV Using 5G NR‑V2X Communication",
IEEE Internet of Things Journal, 2022.

This script:
1) Validates & implements the paper's transmission rate models (cellular Eq. (4), mmWave Eq. (6)-(7)).
2) Implements the paper's MACTER scheme (Algorithms 1 & 2) to maximize computation efficiency (Eq. (17))
   via iterative (near) best-response offloading decisions + bisection-based VEC resource allocation.

Notes / assumptions (kept explicit because the paper leaves some values implicit):
- Power p_i is in Watts. For mmWave SNR in dB, we convert to dBm (as typical when using dB budgets).
- Noise density is set to -174 dBm/Hz, plus a noise figure (NF) in dB, consistent with common practice.
- Link selection: if within mmWave range => mmWave link; else if within cellular range => cellular; else rate=0.
  (The paper motivates distance-based availability; using max(cell,mmW) without range gating can overestimate.)
- e_vec(max) and e_loc(max): used inside the log utility (Eq. (15)-(16)). The paper describes them as
  "maximum tolerable" energy. We implement conservative upper bounds based on configured maxima.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from scipy.stats import truncnorm


# -----------------------------
# Helpers: dB conversions
# -----------------------------
def w_to_dbm(p_w: float) -> float:
    return 10.0 * math.log10(max(p_w, 1e-30) * 1000.0)


def dbm_to_w(p_dbm: float) -> float:
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


# -----------------------------
# Parameters (Table II + paper text)
# -----------------------------
@dataclass(frozen=True)
class SimParams:
    # Network / geometry
    n_vehicles: int = 10
    rsu_range_m: float = 200.0  # r (meters)
    vertical_distance_m: float = 100.0  # e (meters) [paper uses 100 m in sim section]
    mmwave_range_m: float = 150.0
    cellular_range_m: float = 200.0

    # RSU / server
    rsu_max_data_rate_bps: float = 1_000_000_000.0  # γ (bps) cap used in Eq. (10)
    vec_total_cpu_hz: float = 100e9  # F_vec (Hz) total compute at RSU/VEC (choose a reasonable value)

    # Mobility (Eq. (1)-(2))
    speed_mu_kmh: float = 60.0
    speed_sigma_kmh: float = 10.0

    # Cellular link (Eq. (4))
    W_uu_hz: float = 20e6
    cellular_pathloss_exp: float = 3.2  # δ
    h_abs_sq: float = 1.0  # |h|^2 (Rayleigh fading mean ~1)
    noise_density_dbm_per_hz: float = -174.0
    cellular_noise_figure_db: float = 7.0

    # mmWave link (Eq. (6)-(7))
    W_mm_hz: float = 200e6
    mmwave_pathloss_exp: float = 3.2  # ζ
    mmwave_noise_figure_db: float = 7.0
    shadow_fading_db: float = 3.0  # ρ_α (LOS)
    Gmax_vehicle_db: float = 15.0
    Gmax_rsu_db: float = 15.0

    # Task model (simulation section)
    alpha_in_bytes_min: int = 20_000  # 20 kB
    alpha_in_bytes_max: int = 60_000  # 60 kB
    service_coeff_min_cycles_per_byte: float = 0.2e9  # 0.2 GHz per kB -> 0.2e9 cycles per kB
    service_coeff_max_cycles_per_byte: float = 0.4e9  # 0.4e9 cycles per kB

    # Local CPU (simulation section: [10, 15] GHz)
    local_cpu_min_hz: float = 10e9
    local_cpu_max_hz: float = 15e9

    # Energy per computing unit ς_i (Table II shows [2e-7, 2e-6] W per computing unit; we treat as J/cycle scale)
    # We interpret ς_i as Joules per CPU cycle (common in MEC papers), using the same range order.
    energy_per_cycle_min_j: float = 2e-7
    energy_per_cycle_max_j: float = 2e-6

    # Utility / prices (Eq. (15)-(16))
    rho_vec: float = 0.03  # $/(GHz) per Table II (unit price of VEC resource)
    theta_default: float = 0.5  # θ_i
    omega_E_default: float = 0.5  # ϖ_E
    omega_T_default: float = 0.5  # ϖ_T
    phi_penalty: float = 1e6  # ϑ (paper uses to normalize U_loc with indicator penalty); big number works.
    # Bisection
    lambda_min: float = 0.0
    lambda_max: float = 1e3
    eps: float = 1e-6
    max_outer_iters: int = 50


# -----------------------------
# Models from the paper
# -----------------------------
def stay_time(params: SimParams, speed_mps: float) -> float:
    # Eq. (3): t_stay = 2 * sqrt(r^2 - e^2) / v
    r = params.rsu_range_m
    e = params.vertical_distance_m
    return 2.0 * math.sqrt(max(r * r - e * e, 0.0)) / max(speed_mps, 1e-9)


def distance_to_rsu_along_road(params: SimParams, s_i: float) -> float:
    # Paper uses [r * ceil(s_i/r) - s_i] as "distance traveled" factor in (4) and (7).
    r = params.rsu_range_m
    return r * math.ceil(s_i / r) - s_i


def cellular_rate_bps(params: SimParams, p_tx_w: float, d_m: float) -> float:
    # Eq. (4) with σ_uu^2 derived from noise density + NF + bandwidth.
    d = max(d_m, 1e-9)
    noise_dbm = (params.noise_density_dbm_per_hz + params.cellular_noise_figure_db) + 10.0 * math.log10(params.W_uu_hz)
    noise_w = dbm_to_w(noise_dbm)
    snr = (p_tx_w * (d ** (-params.cellular_pathloss_exp)) * params.h_abs_sq) / max(noise_w, 1e-30)
    return params.W_uu_hz * math.log2(1.0 + snr)


def mmwave_rate_bps(params: SimParams, p_tx_w: float, d_m: float) -> float:
    # Eq. (6)-(7): use a dB budget for SNR then convert to linear.
    d = max(d_m, 1e-9)

    p_dbm = w_to_dbm(p_tx_w)
    noise_power_dbm = (params.noise_density_dbm_per_hz + params.mmwave_noise_figure_db) + 10.0 * math.log10(params.W_mm_hz)

    # Pathloss part in dB (matches (7) after rearranging)
    pathloss_db = 10.0 * params.mmwave_pathloss_exp * math.log10(d) + 69.6 + params.shadow_fading_db
    snr_db = (p_dbm - noise_power_dbm) + (params.Gmax_vehicle_db + params.Gmax_rsu_db) - pathloss_db

    snr = db_to_linear(snr_db)
    return params.W_mm_hz * math.log2(1.0 + snr)


def uplink_rate_bps(params: SimParams, p_tx_w: float, d_m: float) -> float:
    """
    Distance-gated link choice:
    - within mmWave range -> mmWave
    - else within cellular range -> cellular
    - else 0
    """
    if d_m <= params.mmwave_range_m:
        rate = mmwave_rate_bps(params, p_tx_w, d_m)
    elif d_m <= params.cellular_range_m:
        rate = cellular_rate_bps(params, p_tx_w, d_m)
    else:
        rate = 0.0

    # γ_i = min{γ (RSU cap), γ_i^uplink}
    return min(rate, params.rsu_max_data_rate_bps)


# -----------------------------
# Task / Vehicle
# -----------------------------
@dataclass
class Task:
    alpha_in_bytes: int  # α_in
    C_cycles: float      # C_i
    t_max_s: float       # t_max


@dataclass
class Vehicle:
    idx: int
    s_pos_m: float
    speed_mps: float
    p_tx_w: float
    f_loc_hz: float
    zeta_j_per_cycle: float
    theta: float
    omega_E: float
    omega_T: float
    task: Task

    # derived
    t_stay_s: float
    t_ptd_s: float
    gamma_bps: float

    # decision vars
    d_loc: int = 1
    d_vec: int = 0
    f_vec_hz: float = 0.0  # allocated by VEC if d_vec=1


def make_vehicles(params: SimParams, seed: int = 0) -> List[Vehicle]:
    random.seed(seed)
    # Truncated Gaussian for speed in km/h then convert to m/s
    v_min, v_max = params.speed_mu_kmh - 3 * params.speed_sigma_kmh, params.speed_mu_kmh + 3 * params.speed_sigma_kmh
    a, b = (v_min - params.speed_mu_kmh) / params.speed_sigma_kmh, (v_max - params.speed_mu_kmh) / params.speed_sigma_kmh
    dist = truncnorm(a, b, loc=params.speed_mu_kmh, scale=params.speed_sigma_kmh)

    vehicles: List[Vehicle] = []
    max_position = params.rsu_range_m * params.n_vehicles  # simplistic road length
    for i in range(params.n_vehicles):
        speed_kmh = float(dist.rvs())
        speed_mps = speed_kmh / 3.6
        s = random.uniform(1.0, max_position)
        d = distance_to_rsu_along_road(params, s)

        p_tx = 1.3  # W (paper sim uses 1.3 W)
        f_loc = random.uniform(params.local_cpu_min_hz, params.local_cpu_max_hz)
        zeta = random.uniform(params.energy_per_cycle_min_j, params.energy_per_cycle_max_j)

        alpha_in = random.randint(params.alpha_in_bytes_min, params.alpha_in_bytes_max)
        # service coefficient in cycles per byte: paper says [0.2,0.4] GHz/kB
        # => [0.2e9,0.4e9] cycles per kB => divide by 1024 for per byte
        mu_cycles_per_kb = random.uniform(params.service_coeff_min_cycles_per_byte, params.service_coeff_max_cycles_per_byte)
        cycles_per_byte = mu_cycles_per_kb / 1024.0
        C = cycles_per_byte * alpha_in

        t_max = random.uniform(0.2, 1.0)  # sim uses [0.2,1] seconds
        task = Task(alpha_in_bytes=alpha_in, C_cycles=C, t_max_s=t_max)

        t_stay = stay_time(params, speed_mps)
        t_ptd = min(t_max, t_stay)

        gamma = uplink_rate_bps(params, p_tx, d)

        vehicles.append(
            Vehicle(
                idx=i,
                s_pos_m=s,
                speed_mps=speed_mps,
                p_tx_w=p_tx,
                f_loc_hz=f_loc,
                zeta_j_per_cycle=zeta,
                theta=params.theta_default,
                omega_E=params.omega_E_default,
                omega_T=params.omega_T_default,
                task=task,
                t_stay_s=t_stay,
                t_ptd_s=t_ptd,
                gamma_bps=gamma,
            )
        )
    return vehicles


# -----------------------------
# Costs, utility, objective (Eq. (8)-(17))
# -----------------------------
def local_time(v: Vehicle) -> float:
    return v.task.C_cycles / max(v.f_loc_hz, 1e-9)  # Eq. (8)


def local_energy(v: Vehicle) -> float:
    return v.zeta_j_per_cycle * v.task.C_cycles  # Eq. (9)


def vec_time(v: Vehicle) -> float:
    # Eq. (10): C/f_vec + alpha_in/gamma
    if v.f_vec_hz <= 0 or v.gamma_bps <= 0:
        return float("inf")
    alpha_bits = 8.0 * v.task.alpha_in_bytes
    return (v.task.C_cycles / v.f_vec_hz) + (alpha_bits / v.gamma_bps)


def vec_energy(v: Vehicle) -> float:
    # Eq. (11): p_i * alpha_in / gamma (alpha_in in bits)
    if v.gamma_bps <= 0:
        return float("inf")
    alpha_bits = 8.0 * v.task.alpha_in_bytes
    return v.p_tx_w * (alpha_bits / v.gamma_bps)


def ETC_local(v: Vehicle) -> float:
    return v.omega_E * local_energy(v) + v.omega_T * local_time(v)  # Eq. (13)


def ETC_vec(v: Vehicle) -> float:
    return v.omega_E * vec_energy(v) + v.omega_T * vec_time(v)  # Eq. (14)


def u_local(v: Vehicle, e_loc_max: float, params: SimParams) -> float:
    # Eq. (15) with indicator penalty
    K = ETC_local(v)
    base = 1.0 + (v.omega_E * e_loc_max + v.omega_T * v.task.t_max_s) - K
    if (v.omega_E * e_loc_max + v.omega_T * v.task.t_max_s) < K:
        return math.log(max(base, 1e-12)) - params.phi_penalty
    return math.log(max(base, 1e-12))


def u_vec(v: Vehicle, e_vec_max: float, params: SimParams) -> float:
    # Eq. (16)
    K = ETC_vec(v)
    base = 1.0 + (v.omega_E * e_vec_max + v.omega_T * v.t_ptd_s) - K
    satisfaction = v.theta * math.log(max(base, 1e-12))
    price_term = (1.0 - v.theta) * params.rho_vec * (v.f_vec_hz / 1e9)  # rho is per GHz
    return max(satisfaction, 0.0) - price_term


def feasibility(v: Vehicle, e_loc_max: float, e_vec_max: float) -> Tuple[bool, bool]:
    # Checks against delay/energy "max" envelopes used in Alg.1 lines 13-14
    Kloc = ETC_local(v)
    Kvec = ETC_vec(v)
    ok_loc = (v.omega_E * e_loc_max + v.omega_T * v.task.t_max_s) >= Kloc
    ok_vec = (v.omega_E * e_vec_max + v.omega_T * v.t_ptd_s) >= Kvec and vec_time(v) <= v.t_ptd_s
    return ok_loc, ok_vec


# -----------------------------
# Algorithm 1: Resource Allocation (Eq. (25)-(26)) + per-vehicle decision
# -----------------------------
def compute_f_vec_from_lambda(v: Vehicle, lam: float, params: SimParams, e_vec_max: float) -> float:
    """
    Implements Algorithm 1 line 4 (Eq. (26)) exactly as written in the paper.
    """
    # a = 1 + (ωE e_vec(max) + ωT t_ptd) - (ωE e_vec + ωT α_in/γ)
    # note: α_in/γ uses bits and bps => seconds.
    alpha_bits = 8.0 * v.task.alpha_in_bytes
    if v.gamma_bps <= 0:
        return 0.0

    a = 1.0 + (v.omega_E * e_vec_max + v.omega_T * v.t_ptd_s) - (v.omega_E * vec_energy(v) + v.omega_T * (alpha_bits / v.gamma_bps))
    denom = (1.0 - v.theta) * params.rho_vec + lam
    denom = max(denom, 1e-12)
    b = - (v.task.C_cycles * v.theta) / denom

    disc = (v.task.C_cycles ** 2) - 4.0 * a * b
    if a <= 0 or disc <= 0:
        return 0.0

    f = (v.task.C_cycles + math.sqrt(disc)) / (2.0 * a)
    # clamp to [0, F] implicitly handled by server constraint; also ensure >0
    return max(f, 0.0)


def allocate_resources_bisection(vehicles: List[Vehicle], params: SimParams) -> None:
    """
    Bisection on lambda so that sum_i f_vec_i ~= F_vec for offloading vehicles.
    Updates v.f_vec_hz.
    """
    offloading = [v for v in vehicles if v.d_vec == 1]
    if not offloading:
        for v in vehicles:
            v.f_vec_hz = 0.0
        return

    # Upper bounds for e_vec(max) per vehicle: use conservative bound based on worst-case rate among connected vehicles.
    # Here: for each vehicle, take current energy and scale by 1.2 to remain feasible (keeps log argument positive).
    e_vec_max = {v.idx: 1.2 * vec_energy(v) for v in offloading}

    lam_min, lam_max = params.lambda_min, params.lambda_max

    for _ in range(200):  # enough for eps ~ 1e-6
        lam = 0.5 * (lam_min + lam_max)
        # compute f for each offloading vehicle
        f_sum = 0.0
        for v in offloading:
            f = compute_f_vec_from_lambda(v, lam, params, e_vec_max[v.idx])
            v.f_vec_hz = f
            f_sum += f

        if (lam_max - lam_min) <= params.eps:
            break

        if f_sum < params.vec_total_cpu_hz:
            lam_max = lam
        else:
            lam_min = lam

    # If sum is far below capacity, keep as-is; if above, scale down proportionally to meet constraint.
    total = sum(v.f_vec_hz for v in offloading)
    if total > params.vec_total_cpu_hz and total > 0:
        scale = params.vec_total_cpu_hz / total
        for v in offloading:
            v.f_vec_hz *= scale

    # Non-offloading vehicles get 0
    for v in vehicles:
        if v.d_vec == 0:
            v.f_vec_hz = 0.0


def update_offloading_decisions(vehicles: List[Vehicle], params: SimParams) -> None:
    """
    Algorithm 1 lines 11-23: compute utilities then choose local vs VEC (or infeasible).
    """
    # e_loc(max): max ς * C
    zeta_max = max(v.zeta_j_per_cycle for v in vehicles)
    e_loc_max = {v.idx: zeta_max * v.task.C_cycles for v in vehicles}

    # e_vec(max): conservative upper bound from current rate (if no rate, infinite)
    e_vec_max = {}
    for v in vehicles:
        if v.gamma_bps <= 0:
            e_vec_max[v.idx] = float("inf")
        else:
            e_vec_max[v.idx] = 1.2 * vec_energy(v)

    for v in vehicles:
        ok_loc, ok_vec = feasibility(v, e_loc_max[v.idx], e_vec_max[v.idx])

        if (not ok_loc) and (not ok_vec):
            v.d_loc, v.d_vec = 0, 0
            continue

        # compute utilities
        Uloc = u_local(v, e_loc_max[v.idx], params) if ok_loc else -float("inf")
        Uvec = u_vec(v, e_vec_max[v.idx], params) if ok_vec else -float("inf")

        if Uvec > Uloc:
            v.d_loc, v.d_vec = 0, 1
        else:
            v.d_loc, v.d_vec = 1, 0


# -----------------------------
# Algorithm 2: Distributed MACTER (best-response style)
# -----------------------------
def computation_efficiency(vehicles: List[Vehicle]) -> float:
    # Eq. (17): sum_i sum_j (phi_i * d_i^j * U_i^j) / E_i
    # We interpret phi_i as computed bits = alpha_in bits.
    total = 0.0
    for v in vehicles:
        alpha_bits = 8.0 * v.task.alpha_in_bytes
        E_i = local_energy(v) + vec_energy(v)  # Eq. (12)
        if not math.isfinite(E_i) or E_i <= 0:
            continue

        # Utilities need current allocated f_vec
        # re-create conservative max bounds for utility computation:
        zeta_max = max(x.zeta_j_per_cycle for x in vehicles)
        e_loc_max = zeta_max * v.task.C_cycles
        e_vec_max = 1.2 * vec_energy(v) if v.gamma_bps > 0 else float("inf")

        Uloc = u_local(v, e_loc_max, params) if v.d_loc == 1 else 0.0
        Uvec = u_vec(v, e_vec_max, params) if v.d_vec == 1 else 0.0

        total += (alpha_bits * (Uloc + Uvec)) / E_i
    return total


def run_macter(params: SimParams, seed: int = 0) -> Tuple[float, List[Vehicle]]:
    vehicles = make_vehicles(params, seed=seed)

    # initial strategy D0 (paper: provided); choose all-local to start
    for v in vehicles:
        v.d_loc, v.d_vec = 1, 0

    prev_D = None
    for _ in range(params.max_outer_iters):
        D = tuple((v.d_loc, v.d_vec) for v in vehicles)
        if D == prev_D:
            break
        prev_D = D

        # Resource allocation given current offloading set
        allocate_resources_bisection(vehicles, params)

        # Update offloading decisions given utilities & constraints
        update_offloading_decisions(vehicles, params)

    # Final allocation for final decisions
    allocate_resources_bisection(vehicles, params)
    E = computation_efficiency(vehicles)
    return E, vehicles


if __name__ == "__main__":
    params = SimParams()
    E, vehicles = run_macter(params, seed=42)
    print(f"Computation efficiency (Eq. 17): {E:.6e}")
    for v in vehicles:
        mode = "VEC" if v.d_vec == 1 else ("LOCAL" if v.d_loc == 1 else "INFEASIBLE")
        print(
            f"veh#{v.idx:02d} mode={mode:10s} gamma={v.gamma_bps/1e6:8.2f} Mbps "
            f"f_loc={v.f_loc_hz/1e9:5.2f} GHz f_vec={v.f_vec_hz/1e9:7.2f} GHz "
            f"t_ptd={v.t_ptd_s:6.3f}s"
        )
