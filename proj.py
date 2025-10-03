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

class Vehicle:
    def __init__(self, speed_distribution):
        self.speed = speed_distribution.rvs()
        self.t_stay = 2 * (math.sqrt((r * r) - (e * e))) / self.speed


v_min, v_max = mu - 3 * sigma, mu + 3 * sigma
a, b = (v_min - mu) / sigma, (v_max - mu) / sigma
dist = truncnorm(a, b, loc=mu, scale=sigma)

vehicles = [Vehicle(dist) for _ in range(n)]
