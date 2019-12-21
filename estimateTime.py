from pulp import *
import math

"""
@author: quanlv
@since: June 13, 2019
@version: 1.0
"""


def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


class EstimateTime:
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

    # #  using Pulp calculate time to charge
    # def calculate(self):
    #     # model LP
    #     m = LpProblem("EstimateTime", LpMaximize)
    #     numNode = len(self.node_pos)
    #     numCharge = len(self.charge_pos)
    #
    #     # variable
    #     x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    #     y = LpVariable.matrix("y", list(range(numNode)), 0, None, LpContinuous)
    #     T = LpVariable("timeSleep", 0, None, LpContinuous)
    #     b = LpVariable.matrix("b", (list(range(numNode)), list(range(numCharge + 1))), 0, 1, LpInteger)
    #
    #     # constrain
    #     for j in range(numNode):
    #         for u in range(numCharge):
    #             m += self.E[j] + lpSum(self.charge(self.node_pos[j], self.charge_pos[i]) * x[i] for i in range(u)) \
    #                  - lpSum((self.time_move[i] + x[i]) * self.e[j] for i in range(u)) >= 10 ** 10 * (y[j] - 1)
    #
    #             m += self.E[j] + lpSum(self.charge(self.node_pos[j], self.charge_pos[i]) * x[i] for i in range(u)) \
    #                  - lpSum((self.time_move[i] + x[i]) * self.e[j] for i in range(u)) <= 10 ** 10 * b[j][u]
    #
    #         m += self.E[j] + lpSum(self.charge(self.node_pos[j], self.charge_pos[i]) * x[i] for i in range(numCharge)) \
    #              - lpSum((self.time_move[i] + x[i]) * self.e[j] for i in range(numCharge)) \
    #              - T * self.e[j] >= 10 ** 10 * (y[j] - 1)
    #
    #         m += self.E[j] + lpSum(self.charge(self.node_pos[j], self.charge_pos[i]) * x[i] for i in range(numCharge)) \
    #              - lpSum((self.time_move[i] + x[i]) * self.e[j] for i in range(numCharge)) \
    #              - T * self.e[j] <= 10 ** 10 * b[j][numCharge]
    #
    #         m += lpSum(b[j][u] for u in range(numCharge + 1)) - numCharge <= y[j]
    #
    #     # objective
    #     m += lpSum(y[j] for j in range(numNode))
    #
    #     # solve problem
    #     status = m.solve()
    #
    #     # return T and stop time at each location
    #     if status != 1:
    #         return -1
    #     else:
    #         temp = []
    #         for i in x:
    #             temp.append(value(i))
    #         return temp, value(T)

    def calculate(self):
        # model LP
        m = LpProblem("EstimateTime", LpMaximize)
        numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)
        E_move = sum(self.time_move) * self.e_move
        # print sum(self.time_move)
        # print E_move
        # variable
        x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
        p = LpVariable.matrix("energy", (list(range(numNode)), list(range(numCharge))), 0, None, LpContinuous)
        target = LpVariable("target", 0, None, LpContinuous)
        T = LpVariable("timeSleep", 0, None, LpContinuous)
        b = LpVariable.matrix("b", (list(range(numNode)), list(range(numCharge))), 0, 1, LpInteger)

        # constrain
        # print len(self.time_move), numCharge
        for j in range(numNode):
            m += target <= self.E[j] - self.e[j] * T - self.time_move[0] * self.e[j]
            for u in range(numCharge):
                m += p[j][u] <= self.charge(self.node_pos[j], self.charge_pos[u])
                m += p[j][u] >= self.charge(self.node_pos[j], self.charge_pos[u]) - 10 ** 10 * (1 - b[j][u])
                m += p[j][u] <= self.E_sensor_max - self.E[j] + (self.time_move[u] + x[u]) * self.e[j]
                m += p[j][u] >= self.E_sensor_max - self.E[j] + (self.time_move[u] + x[u]) * self.e[j] - 10 ** 10 * \
                     b[j][u]
                # m += p[j][u] == min(self.charge(self.node_pos[j], self.charge_pos[u]),
                #                     self.E_sensor_max - self.E[j] + (self.time_move[u] + x[u]) * self.e[j])
                m += target <= self.E[j] - self.e[j] * T + lpSum([p[j][i] for i in range(u)]) - lpSum(
                    [self.time_move[i] + x[i] for i in range(u)]) * self.e[j] - self.time_move[u + 1] * self.e[j]
        m += lpSum(p[j][u] for j in range(numNode) for u in range(numCharge)) + E_move <= self.E_mc + T * self.e_mc
        m += self.E_mc + T * self.e_mc <= self.E_mc_max

        # objective
        m += target

        # solve problem
        status = m.solve()

        # return T and stop time at each location
        if status != 1:
            print status
            return -1
        else:
            temp = []
            for i in x:
                temp.append(value(i))
            # print value(T), temp
            return temp, value(T)

    # def calculate(self):
    #     # model LP
    #     m = LpProblem("EstimateTime", LpMaximize)
    #     numNode = len(self.node_pos)
    #     numCharge = len(self.charge_pos)
    #     E_move = sum(self.time_move) * self.e_move
    #     print sum(self.time_move)
    #     # print E_move
    #     # variable
    #     x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
    #     # p = LpVariable.matrix("energy", (list(range(numNode)), list(range(numCharge))), 0, None, LpContinuous)
    #     target = LpVariable("target", 0, None, LpContinuous)
    #     T = LpVariable("timeSleep", 0, None, LpContinuous)
    #     # b = LpVariable.matrix("b", (list(range(numNode)), list(range(numCharge))), 0, 1, LpInteger)
    #
    #     # constrain
    #     # print len(self.time_move), numCharge
    #     for j in range(numNode):
    #         m += target <= self.E[j] - self.e[j] * T - self.time_move[0] * self.e[j]
    #         for u in range(numCharge):
    #             m += target <= self.E[j] - self.e[j] * T + lpSum(
    #                 self.charge(self.node_pos[j], self.charge_pos[i]) for i in range(u)) - lpSum(
    #                 [self.time_move[i] + x[i] for i in range(u)]) * self.e[j] - self.time_move[u + 1] * self.e[j]
    #     m += lpSum(self.charge(self.node_pos[j], self.charge_pos[u]) for j in range(numNode) for u in
    #                range(numCharge)) <= self.E_mc + T * self.e_mc - E_move
    #     m += self.E_mc + T * self.e_mc <= self.E_mc_max
    #     m += self.E_mc + T * self.e_mc >= E_move
    #
    #     # objective
    #     m += target
    #
    #     # solve problem
    #     status = m.solve()
    #
    #     # return T and stop time at each location
    #     if status != 1:
    #         print status
    #         return -1
    #     else:
    #         temp = [type(item) if value(item) else 0.0 for item in x]
    #         print value(target)
    #         # print value(T), temp
    #         return temp, value(T)
