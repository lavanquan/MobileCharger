from pulp import *
import math

"""
@author: quanlv
@since: June 13, 2019
@version: 1.0
"""


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


class EstimateTimeINMA:
    # class estimate time of mobile charger at every charge location
    def __init__(self, E, e, E_sensor_max, E_mc, e_mc, e_move, E_mc_max, node_pos, charge_pos, time_move,
                 chargeRange=50, alpha=50,
                 beta=5):
        self.E = E  # energy of sensor
        self.e = e  # average used energy of sensor
        self.E_mc = E_mc  # energy of MC
        self.e_mc = e_mc  # charge energy per second of mc
        self.e_move = e_move  # energy to move of mc
        self.E_mc_max = E_mc_max  # max energy of mc
        self.node_pos = node_pos  # location of sensor
        self.charge_pos = charge_pos  # location of charge
        self.time_move = time_move  # time move of mc
        self.chargeRange = chargeRange  # radius of communication
        self.alpha = alpha  # charge parameter
        self.beta = beta  # charge parameter
        self.E_sensor_max = E_sensor_max  # max energy of mc

    # charge per second of mobile charger
    def charge(self, node, charge):
        d = distance(node, charge)
        if d > self.chargeRange:
            return 0
        else:
            return self.alpha / ((d + self.beta) ** 2)

    def calculate(self):
        return 0
