from pulp import *
import math

"""
@author: Quan La Van
@since: June 13, 2019
@version: 1.0
"""


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


class EstimateTimeGreedy:
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

    def getWeight(self, gamma):
        numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)

        model = LpProblem("Charge", LpMinimize)
        x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
        t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
        for j, _ in enumerate(self.node_pos):
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) - gamma * (self.e[j] / sum(self.e)) + (
                             1 - gamma) * (self.E[j] / sum(self.E)) <= t[j]
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) - gamma * (self.e[j] / sum(self.e)) + (
                             1 - gamma) * (self.E[j] / sum(self.E)) >= -t[j]
        model += lpSum(t)
        status = model.solve()
        if status == 1:
            valueX = [value(item) for item in x]
            if sum(valueX):
                w = [item / sum(valueX) for item in valueX]
            else:
                w = [1.0 / len(self.charge_pos) for _ in self.charge_pos]
            return w
        else:
            print "LP can not solve"
            return -1

    def getWeight2(self, gamma):
        numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)

        model = LpProblem("Charge", LpMinimize)
        x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
        t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
        for j, _ in enumerate(self.node_pos):
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) + (self.E[j] / sum(self.E)) <= t[j]
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) + (self.E[j] / sum(self.E)) >= -t[j]
        model += lpSum(t)
        status = model.solve()
        if status == 1:
            valueX = [value(item) for item in x]
            if sum(valueX):
                w = [item / sum(valueX) for item in valueX]
            else:
                w = [1.0 / len(self.charge_pos) for _ in self.charge_pos]
            return w
        else:
            print "LP can not solve"
            return -1

    def getWeight3(self, gamma):
        numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)

        model = LpProblem("Charge", LpMinimize)
        x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
        t = LpVariable.matrix("t", list(range(numNode)), 0, None, LpContinuous)
        for j, _ in enumerate(self.node_pos):
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) - (self.e[j] / sum(self.e)) <= t[j]
            model += lpSum([x[u] * (self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j]) for u, _ in
                            enumerate(self.charge_pos)]) / lpSum(
                [self.charge(self.node_pos[j], self.charge_pos[u]) - self.e[j] for u, _ in
                 enumerate(self.charge_pos)]) - (self.e[j] / sum(self.e)) >= -t[j]
        model += lpSum(t)
        status = model.solve()
        if status == 1:
            valueX = [value(item) for item in x]
            if sum(valueX):
                w = [item / sum(valueX) for item in valueX]
            else:
                w = [1.0 / len(self.charge_pos) for _ in self.charge_pos]
            return w
        else:
            print "LP can not solve"
            return -1

    def calculate(self, gamma):
        # numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)
        w = self.getWeight(gamma=gamma)
        if w == -1:
            return -1

        maxP = max(
            sum(self.charge(self.node_pos[j], self.charge_pos[u]) for j, _ in enumerate(self.node_pos)) for u, _ in
            enumerate(self.charge_pos))
        sumP = sum(
            sum(self.charge(self.node_pos[j], self.charge_pos[u]) for j, _ in enumerate(self.node_pos)) for u, _ in
            enumerate(self.charge_pos))

        model = LpProblem("getTime", LpMaximize)

        x = LpVariable("x", 0, None, LpContinuous)  # energy
        t_k = LpVariable("t_k", 0, None, LpContinuous)  # time charge at base
        E_mc = LpVariable("E_mc", 0, self.E_mc_max, LpContinuous)  # energy of MC after charge
        T_base = LpVariable("T_base", 0, None, LpContinuous)  # time minimum to charge
        T = LpVariable.matrix("T", list(range(numCharge)), 0, None, LpContinuous)  # time charge at each location

        model += E_mc + t_k * self.e_mc >= lpSum(self.time_move) * self.e_move + lpSum(
            [self.E_sensor_max - self.E[j] for j, _ in enumerate(self.node_pos)])
        model += E_mc == self.E_mc + t_k * self.e_mc - self.e_move * lpSum(self.time_move)
        model += T_base == E_mc / maxP
        for u, _ in enumerate(self.charge_pos):
            model += T[u] == T_base * w[u]
        for j, _ in enumerate(self.node_pos):
            model += x <= self.E[j] - t_k * self.e[j] - self.time_move[0] * self.e[j]
            for u, _ in enumerate(self.charge_pos):
                model += x <= self.E[j] - t_k * self.e[j] + lpSum(
                    (self.charge(self.node_pos[j], self.charge_pos[i]) - self.e[j]) * T[i] for i in range(u)) - lpSum(
                    self.time_move[i] + T[i] for i in range(u)) * self.e[j] - self.time_move[u + 1] * self.e[j]

        model += x

        status = model.solve()

        if status == 1:
            valueX = [value(item) for item in T]
            return valueX, value(t_k)
        else:
            return -1

    # def calculate(self, gamma):
    #     w = self.getWeight(gamma=gamma)
    #     E_lose = sum([self.E_sensor_max - self.E[j] for j, _ in enumerate(self.node_pos)])
    #     print E_lose, sum(self.time_move) * self.e_move
    #     if self.E_mc < E_lose + sum(self.time_move) * self.e_move < self.E_mc_max:
    #         E_new = E_lose + sum(self.time_move) * self.e_move
    #     elif self.E_mc >= E_lose + sum(self.time_move) * self.e_move:
    #         E_new = self.E_mc
    #     else:
    #         E_new = self.E_mc_max
    #     E_new = self.E_mc_max
    #     t_k = (E_new - self.E_mc) / self.e_mc
    #     t_base = (E_new - sum(self.time_move) * self.e_move) / (sum([sum([self.charge(node, charge) for j, node in enumerate(self.node_pos)]) * w[u] for u, charge in enumerate(self.charge_pos)]))
    #     x = [w[u] * t_base for u, _ in enumerate(w)]
    #     return x, t_k

    # def calculate(self, gamma):
    #     # numNode = len(self.node_pos)
    #     numCharge = len(self.charge_pos)
    #     w = self.getWeight3(gamma=gamma)
    #     E_lose = sum([self.E_sensor_max - self.E[j] for j, _ in enumerate(self.node_pos)])
    #     if self.E_mc < E_lose + sum(self.time_move) * self.e_move < self.E_mc_max:
    #         E_new = E_lose + sum(self.time_move) * self.e_move
    #     elif self.E_mc >= E_lose + sum(self.time_move) * self.e_move:
    #         E_new = self.E_mc
    #     else:
    #         E_new = self.E_mc_max
    #     t_k = (E_new - self.E_mc) / self.e_mc
    #     if w == -1:
    #         return -1
    #
    #     maxP = max(
    #         sum(self.charge(self.node_pos[j], self.charge_pos[u]) for j, _ in enumerate(self.node_pos)) for u, _ in
    #         enumerate(self.charge_pos))
    #     sumP = sum(
    #         sum(self.charge(self.node_pos[j], self.charge_pos[u]) for j, _ in enumerate(self.node_pos)) for u, _ in
    #         enumerate(self.charge_pos))
    #
    #     model = LpProblem("getTime", LpMaximize)
    #
    #     x = LpVariable("x", 0, None, LpContinuous)  # energy
    #     T_base = LpVariable("T_base", 0, None, LpContinuous)  # time minimum to charge
    #     T = LpVariable.matrix("T", list(range(numCharge)), 0, None, LpContinuous)  # time charge at each location
    #
    #     for u, _ in enumerate(self.charge_pos):
    #         model += T[u] == T_base * w[u]
    #     for j, _ in enumerate(self.node_pos):
    #         model += x <= self.E[j] - t_k * self.e[j] - self.time_move[0] * self.e[j]
    #         for u, _ in enumerate(self.charge_pos):
    #             model += x <= self.E[j] - t_k * self.e[j] + lpSum(
    #                 (self.charge(self.node_pos[j], self.charge_pos[i]) - self.e[j]) * T[i] for i in range(u)) - lpSum(
    #                 self.time_move[i] + T[i] for i in range(u)) * self.e[j] - self.time_move[u + 1] * self.e[j]
    #
    #     model += x
    #
    #     status = model.solve()
    #
    #     if status == 1:
    #         valueX = [value(item) for item in T]
    #         return valueX, t_k
    #     else:
    #         return -1
