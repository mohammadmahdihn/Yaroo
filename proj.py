import numpy as np
from scipy.optimize import minimize
import math
import random
from scipy.stats import truncnorm

n = 10  # number of vehicles
r = 200  # RSU communication range
e = 160  # vertical distance between the road and RSU
mu = 60  # average speed
sigma = 10  # standard deviation of speed
mm_wave_comm_range = 150
cellular_link_comm_range = 200
max_position = r * n

# cellular transmission rate variables
W_uu = 20_000_000  # cellular channel bandwidth
noise_density_dbm_per_hz = -174.0
noise_figure_db = 7.0
h_abs_sq = 1.0  # |h^2|
cellular_path_loss_exponent = 3.2

W_mm = 200_000_000  # mmWave channel bandwidth

class Vehicle:
    def __init__(self, speed_distribution):
        self.speed = speed_distribution.rvs()
        self.t_stay = 2 * (math.sqrt((r * r) - (e * e))) / self.speed
        self.s = random.randint(1, max_position)  # s_i (current position)
        self.p_i = 1.3  # Transmission power of vehicle
        self.transmission_rate = self.calculate_transmission_rate()

    def calculate_transmission_rate(self):
        distance = r * math.ceil(self.s / r) - self.s
        cellular = self.calculate_cellular_transmission_rate(distance)
        mmWave = self.calculate_mm_wave_transmission_rate(distance)
        return

    def calculate_cellular_transmission_rate(self, distance):
        d = max(distance, 1e-9)

        noise_dbm = noise_density_dbm_per_hz + noise_figure_db + 10.0 * math.log10(W_uu)
        noise_w = 10.0 ** ((noise_dbm - 30.0) / 10.0)  # dBm -> W

        snr = (self.p_i * (d ** (-cellular_path_loss_exponent)) * h_abs_sq) / noise_w
        return W_uu * math.log2(1.0 + snr)

    def calculate_mmWave_transmission_rate(self, distance):
        pass


v_min, v_max = mu - 3 * sigma, mu + 3 * sigma
a, b = (v_min - mu) / sigma, (v_max - mu) / sigma
dist = truncnorm(a, b, loc=mu, scale=sigma)

vehicles = [Vehicle(dist) for _ in range(n)]
