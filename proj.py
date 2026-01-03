import math
import random
from scipy.stats import truncnorm

n = 10  # number of vehicles
r = 200  # RSU communication range
e = 120  # vertical distance between the road and RSU
r_2 = 2 * math.sqrt(r ** 2 - e ** 2)  # Length of one RSU coverage segment
mu = 60  # average speed
sigma = 10  # standard deviation of speed
mm_wave_comm_range = 150
cellular_link_comm_range = 200
max_position = r_2 * n
rsu_max_data_rate = 1_000_000_000

# cellular transmission rate variables
W_uu = 20_000_000  # cellular channel bandwidth
cellular_noise_density_dbm_per_hz = -174.0
cellular_noise_figure_db = 7.0
h_abs_sq = 1.0  # |h^2|
cellular_path_loss_exponent = 3.2

# mmWave transmission rate variables
W_mm = 200_000_000  # mmWave channel bandwidth
mmWave_noise_density_dbm_per_hz = -174.0
mmWave_noise_figure_db = 7.0
mmWave_path_loss_exponent = 3.2
shadow_fading_db = 3.0  # ρα
Gmax_vehicle_db = 15.0  # RSU main-lobe gain (dB)
Gmax_rsu_db = 15.0  # Vehicle main-lobe gain (dB)

def w_to_dbm(p_w):
    return 10.0 * math.log10(p_w * 1000.0)

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)

class Task:
    def __init__(self):
        self.h = 2
        self.alpha_in = random.randint(4000, 20000)
        self.C = self.h * self.alpha_in
        self.t_max = random.randint(5, 10)

class Vehicle:
    def __init__(self, speed_distribution):
        self.speed = speed_distribution.rvs()
        self.t_stay = 2 * (math.sqrt((r * r) - (e * e))) / self.speed
        self.s = random.randint(1, round(max_position))  # s_i (current position)
        self.p_i = 1.3  # Transmission power of vehicle
        self.transmission_rate = self.calculate_transmission_rate()
        self.task = Task()
        self.t_ptd = min(self.task.t_max, self.t_stay)

    def calculate_transmission_rate(self):
        distance_to_end_of_rsu_range = r_2 * math.ceil(self.s / r_2) - self.s
        distance_to_rsu_center = math.sqrt((abs((r_2/2) - distance_to_end_of_rsu_range) ** 2) + (e ** 2))
        if distance_to_rsu_center <= mm_wave_comm_range:
            rate = self.calculate_cellular_transmission_rate(distance_to_end_of_rsu_range)
        elif distance_to_rsu_center <= cellular_link_comm_range:
            rate = self.calculate_mmWave_transmission_rate(distance_to_end_of_rsu_range)
        else:
            raise Exception
        return min(rate, rsu_max_data_rate)

    def calculate_cellular_transmission_rate(self, distance):
        d = max(distance, 1e-9)

        noise_dbm = cellular_noise_density_dbm_per_hz + cellular_noise_figure_db + 10.0 * math.log10(W_uu)
        noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)  # dBm -> W

        snr = (self.p_i * (d ** (-cellular_path_loss_exponent)) * h_abs_sq) / noise_w
        return W_uu * math.log2(1.0 + snr)

    def calculate_mmWave_transmission_rate(self, distance):
        d = max(distance, 1e-9)

        p_dbm = w_to_dbm(self.p_i)
        noise_power_dbm = (mmWave_noise_density_dbm_per_hz + mmWave_noise_figure_db) + 10.0 * math.log10(W_mm)

        path_loss_db = 10.0 * mmWave_path_loss_exponent * math.log10(d) + 69.6 + shadow_fading_db
        snr_db = (p_dbm - noise_power_dbm) + (Gmax_vehicle_db + Gmax_rsu_db) - path_loss_db

        snr = db_to_linear(snr_db)
        return W_mm * math.log2(1.0 + snr)


v_min, v_max = mu - 3 * sigma, mu + 3 * sigma
a, b = (v_min - mu) / sigma, (v_max - mu) / sigma
dist = truncnorm(a, b, loc=mu, scale=sigma)

vehicles = [Vehicle(dist) for _ in range(n)]