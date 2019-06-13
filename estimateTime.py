from pulp import *
import math

"""
@author: quanlv
@since: June 13, 2019
@version: 1.0
"""
class EstimateTime:
    # class uoc luong thoi gian sac cua mobile charger tai moi vi tri sac
    def __init__(self, E, e, node_pos, charge_pos, time_move, chargeRange=50, alpha=50, beta=5):
        self.E = E # nang luong con lai cua moi sensor
        self.e = e # nang luong tieu thu trung binh trong mot don vi thoi gian
        self.node_pos = node_pos # toa do cam bien
        self.charge_pos = charge_pos # toa do charge
        self.time_move = time_move #thoi gian di chuyen cua charger
        self.chargeRange = chargeRange #ban kinh sac
        self.alpha = alpha # he so sac
        self.beta = beta # he so sac

    def distance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

    # nang luong sac moi giay cua mobile charger
    def charge(self, node, charge):
        d = self.distance(node, charge)
        if d > self.chargeRange:
            return 0
        else:
            return self.alpha / ((d+self.beta)**2)

    #  tinh toan thoi gian sac, su dung model cua pulp
    def calculate(self):
        # model quy hoach nguyen
        m = LpProblem("EstiamteTime", LpMaximize)
        numNode = len(self.node_pos)
        numCharge = len(self.charge_pos)

        # dinh nghia cac bien so
        x = LpVariable.matrix("x", list(range(numCharge)), 0, None, LpContinuous)
        y = LpVariable.matrix("y", list(range(numNode)), 0, None, LpContinuous)
        T = LpVariable("timeSleep", 0, None, LpContinuous)
        b = LpVariable.matrix("b", (list(range(numNode)), list(range(numCharge + 1))), 0, 1, LpInteger)

        # dinh nghia cac rang buoc
        for j in range(numNode):
            for u in range(numCharge):
                m += self.E[j] + lpSum(self.charge(self.node_pos[j],self.charge_pos[i]) * x[i] for i in range(u)) \
                     - lpSum((self.time_move[i]+x[i])*self.e[j] for i in range(u)) >= 10**10 * (y[j]-1)

                m += self.E[j] + lpSum(self.charge(self.node_pos[j],self.charge_pos[i]) * x[i] for i in range(u)) \
                     - lpSum((self.time_move[i]+x[i])*self.e[j] for i in range(u)) <= 10**10 * b[j][u]

            m += self.E[j] + lpSum(self.charge(self.node_pos[j],self.charge_pos[i]) * x[i] for i in range(numCharge)) \
                 - lpSum((self.time_move[i]+x[i]) * self.e[j] for i in range(numCharge)) \
                 - T * self.e[j] >= 10**10 *(y[j]-1)

            m += self.E[j] + lpSum(self.charge(self.node_pos[j],self.charge_pos[i]) * x[i] for i in range(numCharge)) \
                 - lpSum((self.time_move[i]+x[i]) * self.e[j] for i in range(numCharge)) \
                 - T * self.e[j] <= 10**10 * b[j][numCharge]

            m += lpSum(b[j][u] for u in range(numCharge+1)) - numCharge <= y[j]

        # ding nghia objective
        m += lpSum(y[j] for j in range(numNode))

        # solve problem
        status  = m.solve()

        # return list thoi gian sac tai moi vi tri va T - thoi gian mobile charger nghi tai base
        if status != 1:
            return -1
        else:
            temp = []
            for i in x:
                temp.append(value(i))
            return temp, value(T)