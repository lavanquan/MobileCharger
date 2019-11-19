from pymote import *
import pandas as pd
from ast import literal_eval
import math
import csv
from estimateTime import EstimateTime
from estimateTimeGreedy import EstimateTimeGreedy
import random

"""
@author: quanlv
@since: June 13, 2019
@version: 1.0
"""


#  distance from 2 location
def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


class NetworkSimulator:
    # doc cac tham so tu file data, index de chi bo du lieu, trong file data - moi dong la 1 bo du lieu
    def __init__(self, file_name="data.csv", index=0):
        df = pd.read_csv(file_name)

        # cac tham so cua sensor
        self.node_pos = list(literal_eval(df.node_pos[index]))  # vi tri cac node
        self.numNode = len(self.node_pos)  # so luong node
        self.energy = df.energy[index]  # nang luong max
        self.E_sensor_max = df.energy[index]
        self.commRange = df.commRange[index]  # ban kinh truyen
        self.freq = df.freq[index]  # ti le gui tin
        # self.prob = [int(item) for item in df.prob[index].split(",")]  # tan xuat sinh goi tin cua cac node

        # cac tham so cua mobile charger
        self.charge_pos = list(literal_eval(df.charge_pos[index]))  # vi tri sac
        self.numCharge = len(self.charge_pos)  # so luong vi tri mobile charger dung chan
        self.chargeRange = df.chargeRange[index]  # ban kinh sac
        self.velocity = df.velocity[index]  # van toc mobile charger

        self.base = literal_eval(df.base[index])  # toa do cua base

        self.ER = df.ER[index]  # pra tinh nang luong truyen thong
        self.ET = df.ET[index]  # pra tinh nang luong truyen thong
        self.EFS = df.EFS[index]  # pra tinh nang luong truyen thong
        self.EMP = df.EMP[index]  # pra tinh nang luong truyen thong

        self.b = df.b[index]  # kich thuoc goi tin can bien
        self.b_energy = df.b_energy[index]  # kich thuoc goi tin nang luong
        self.THRESHOLD = df.THRESHOLD[index]  # nguong nang luong toi thieu de sensor con song
        self.delta = df.delta[index]  # thoi gian dinh ki gui goi tin nang luong

        self.alpha = df.alpha[index]  # pra tinh nang luong sac
        self.beta = df.beta[index]  # pra tinh nang luong sac

        self.MaxTime = df.MaxTime[index]  # thoi gian chay mo phong toi da

        self.E_mc = df.E_mc[index]
        self.E_mc_max = df.E_max[index]
        self.e_mc = df.e_mc[index]
        self.e_move = df.e_move[index]

        # network
        self.net = Network()
        idx = 0
        for nd in self.node_pos:
            self.net.add_node(pos=nd, commRange=self.commRange, energy=self.energy)
            idx = idx + 1

    # Xac dinh node se nhan goi tin quang ba
    def destinate(self, node):
        minDis = distance(self.net.pos[node], self.base)
        idNode = -1
        if distance(self.net.pos[node], self.base) <= self.commRange:
            idNode = self.base
        else:
            for nd in self.net.neighbors(node):
                d = distance(self.net.pos[nd], self.base)
                if d < minDis:
                    minDis = d
                    idNode = nd
        return idNode

    # Xac dinh duong di cua goi tin tu node den base
    def rounte(self, node):
        path = [node]
        if distance(self.net.pos[node], self.base) <= self.commRange:
            return path
        else:
            temp = self.destinate(node)
            if temp != -1:
                path.extend(self.rounte(temp))
            return path

    # Nang luong mat mat khi gui goi tin
    def sendNode(self, send, receive, energy):
        d0 = math.sqrt(self.EFS / self.EMP)
        d = 0
        if receive == self.base:
            d = distance(self.net.pos[send], self.base)
        else:
            d = distance(self.net.pos[send], self.net.pos[receive])
        e_send = 0
        if d < d0:
            e_send = self.ET + self.EFS * d ** 2
        else:
            e_send = self.ET + self.EMP * d ** 4

        send.energy -= e_send * energy

    # Nang luong mat mat khi nhan goi tin
    def receiveNode(self, receive, energy):
        receive.energy = receive.energy - self.ER * energy

    # Kiem tra node con song hay da chet
    def isDead(self, node):
        if node.energy < self.THRESHOLD:
            return True
        else:
            return False

    # Ham gui tin tu mot node bat ki den base
    # Sau nay chi can dung ham nay de mo phong qua trinh gui tin khi co tinh den yeu to nang luong
    def send(self, node, energy):
        for nd in self.net.neighbors(node):
            if self.isDead(nd):
                self.net.remove_node(nd)
        des = self.destinate(node)
        if des != -1:
            self.sendNode(node, des, energy)
            for nd in self.net.neighbors(node):
                self.receiveNode(nd, energy)
            if des != self.base:
                self.send(des, energy)

    # Ghi file
    def writeHeader(self, file_name="log_file.csv"):
        header = []
        for node in self.net.nodes():
            header.append(str(node.id))
        with open(file_name, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()
        csv_file.close()

    # Ghi file
    def write(self, file_name="log_file.csv"):
        header = []
        for node in self.net.nodes():
            header.append(str(node.id))
        with open(file_name, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            row = {}
            for node in self.net.nodes():
                if self.isDead(node):
                    row[str(node.id)] = 0
                else:
                    row[str(node.id)] = node.energy
            writer.writerow(row)
        csv_file.close()

    # Thuc hien truyen tin trong mang tai thoi diem t voi tham so gui goi tin nang luong delta
    # Sau nay chi can goi ham communicate() thi toan mang se duoc thuc hien gui tin
    def communicateUniform(self, t, delta, freq):
        E_old = [node.energy for node in self.net.nodes()]
        for node in self.net.nodes():
            r = random.random()
            if r <= freq:
                self.send(node, self.b)
        if t % delta == 0:
            for node in self.net.nodes():
                self.send(node, self.b_energy)
        E_new = [node.energy for node in self.net.nodes()]
        if len(E_new) != self.numNode:
            return -1
        e = [E_old[index] - E_new[index] for index in range(self.numNode)]
        return e

    # nang luong sac moi giay cua mobile charger cho toan mang
    # t: thoi diem hien tai
    # timeStart: thoi diem bat dau hanh trinh cua mobile charger
    # chare_time: danh sach thoi gian dung chan cua mobile charger
    # time_move: thoi gian mobile charger di chuyen qua cac diem dung chan
    def chargePerSec(self, t, timeStart, charge_time, time_move):
        index = -1
        for i in range(len(charge_time)):
            if timeStart + sum(time_move[:i + 1]) + sum(charge_time[:i]) < t <= timeStart + sum(
                    time_move[:i + 1]) + sum(charge_time[:i + 1]):
                index = i
        if index != -1:
            E_old = [node.energy for node in self.net.nodes()]
            start = timeStart + sum(time_move[:index + 1]) + sum(charge_time[:index])
            end = start + charge_time[index]
            if t - start < 1:
                for node in self.net.nodes():
                    d = distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        p = min(self.alpha / ((d + self.beta) ** 2) * (t - start), self.E_sensor_max - node.energy)
                        node.energy += p
            elif end - t < 1:
                for node in self.net.nodes():
                    d = distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        p = min(self.alpha / ((d + self.beta) ** 2) * (end - t), self.E_sensor_max - node.energy)
                        node.energy += p
            else:
                for node in self.net.nodes():
                    d = distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        p = min(self.alpha / ((d + self.beta) ** 2), self.E_sensor_max - node.energy)
                        node.energy += p
            E_new = [node.energy for node in self.net.nodes()]
            # print (sum(E_new) - sum(E_old))
            self.E_mc = self.E_mc - (sum(E_new) - sum(E_old))

    # estimate E and e using average
    # list_energy: the average of used energy
    # num_point: the number of point take to calculate
    def estimateEnergy(self, list_energy, num_point):
        top = [0.0 for i in range(self.numNode)]
        bot = [0.0 for i in range(self.numNode)]
        e = [0.0 for i in range(self.numNode)]
        if num_point < len(list_energy):
            n = num_point
        else:
            n = len(list_energy)
        for idNode in range(self.numNode):
            for index in range(n):
                energy_index = list_energy[len(list_energy) - 1 - index]
                top[idNode] = top[idNode] + energy_index[idNode][0] * energy_index[idNode][1]
                bot[idNode] = bot[idNode] + energy_index[idNode][1]
            e[idNode] = top[idNode] / bot[idNode]
        E = []
        for nd in self.net.nodes():
            E.append(nd.energy)
        return E, e

    def getIndexMinNode(self):
        min_e = 100
        index = 0
        for i, node in enumerate(self.net.nodes()):
            if node.energy < min_e:
                min_e = node.energy
                index = i
        return index, round(min_e, 3)

    # ham thuc hien mo phong qua trinh mang thuc thi
    # file_name la ten file se log du lieu nang luong cua mang
    def sim(self, file_name="log_file.csv", num_point=100):
        t = 1
        self.writeHeader(file_name)

        # timeStart: thoi gian bat dau sac
        # timeStop: thoi gian ket thuc sac
        timeStart = self.delta
        timeStop = self.delta
        flag = False

        # dis_charge_pos = sum([self.distance(self.charge_pos[i], self.charge_pos[i + 1]) \
        #                       for i in range(len(self.charge_pos) - 1)])
        # calculate time to move
        charge_pos_extend = [(0, 0)]
        charge_pos_extend.extend(self.charge_pos)
        charge_pos_extend.extend([(0, 0)])
        time_move = [distance(charge_pos_extend[i], charge_pos_extend[i + 1]) / self.velocity \
                     for i in range(len(charge_pos_extend) - 1)]
        print self.e_move * sum(time_move)
        # list of average used energy
        list_energy = []
        # temp_e is used to save the used energy each delta second
        temp_e = [0.0 for _ in range(self.numNode)]
        charge_add74 = 0
        e74 = 0.0
        while t <= self.MaxTime:
            # print t, min([node.energy for node in self.net.nodes()]), max([node.energy for node in self.net.nodes()])
            # temp_e1 is used to save the used energy each second
            temp_e1 = self.communicateUniform(t, self.delta, freq=self.freq)
            if temp_e1 == -1:
                break
            else:
                temp_e = [temp_e[index] + temp_e1[index] for index in range(self.numNode)]
            # if t % delta = 0, network will log used energy information to the base.
            if t % self.delta == 0:
                temp_list_energy = [(temp_e[index] / self.delta, t) for index in range(self.numNode)]
                temp_e = [0.0 for _ in range(self.numNode)]
                list_energy.append(temp_list_energy)

            if t == self.delta:
                E, e = self.estimateEnergy(list_energy=list_energy, num_point=num_point)
                e74 = e[74]
                print e[74], E[74], self.E_mc, self.E_mc_max
                estimate = EstimateTimeGreedy(E=E, e=e, E_sensor_max=self.E_sensor_max, E_mc=self.E_mc, e_mc=self.e_mc,
                                              e_move=self.e_move,
                                              E_mc_max=self.E_mc_max, node_pos=self.node_pos,
                                              charge_pos=self.charge_pos,
                                              time_move=time_move, chargeRange=self.chargeRange,
                                              alpha=self.alpha, beta=self.beta)
                status = estimate.calculate(0.5)
                if status == -1:
                    t = t + 1
                    continue
                charge_time, T = status
                self.E_mc = self.E_mc + T * self.e_mc
                print T, charge_time

                timeStart = timeStop + T
                timeStop = timeStart + sum(time_move) + sum(charge_time)

            if t == math.floor(timeStart):
                flag = True
            elif t == math.floor(timeStop):
                flag = False
                E, e = self.estimateEnergy(list_energy=list_energy, num_point=num_point)
                print min(e), max(e), min(E), max(E), self.E_mc, self.E_mc_max
                estimate = EstimateTimeGreedy(E=E, e=e, E_sensor_max=self.E_sensor_max, E_mc=self.E_mc, e_mc=self.e_mc,
                                              e_move=self.e_move,
                                              E_mc_max=self.E_mc_max, node_pos=self.node_pos,
                                              charge_pos=self.charge_pos,
                                              time_move=time_move, chargeRange=self.chargeRange,
                                              alpha=self.alpha, beta=self.beta)
                status = estimate.calculate(0.5)
                if status == -1:
                    t = t + 1
                    continue
                charge_time, T = status
                self.E_mc = self.E_mc + T * self.e_mc
                print T, charge_time

                timeStart = timeStop + T + 1
                timeStop = timeStart + sum(time_move) + sum(charge_time)
            node74_old = self.net.nodes()[74].energy
            if flag:
                # if self.E_mc < sum(time_move) * self.e_move:
                #     flag = False
                self.chargePerSec(t, timeStart, charge_time, time_move)
            # if t % self.delta == 0:
            #     self.write(file_name)
            node74_new = self.net.nodes()[74].energy
            # print "t = ", t, self.getIndexMinNode(), round(min([node.energy for node in self.net.nodes()]), 3), round(
            #     max([node.energy for node in self.net.nodes()]), 3)
            charge_add74 += node74_new - node74_old
            # if node74_new - node74_old > 0:
            #     print "t = ", t, node74_new - node74_old
            t += 1
        print t

    def simNoCharge(self, file_name="log_file.csv", num_point=100):
        t = 1
        while t <= self.MaxTime:
            # if t == 100:
            #     min_e = 100
            #     index = 0
            #     for i, node in enumerate(self.net.nodes()):
            #         if node.energy < min_e:
            #             min_e = node.energy
            #             index = i
            #     print "min energy at ", index, min_e
            # print t, min([node.energy for node in self.net.nodes()]), max([node.energy for node in self.net.nodes()])
            temp_e1 = self.communicateUniform(t, self.delta, freq=self.freq)
            if temp_e1 == -1:
                break
            t = t + 1
        print t

    def getNodeDead(self, e, thread, mc_location):
        E_thread = thread * self.E_sensor_max
        list_request = [index for index, nodeId in enumerate(self.net.nodes()) if nodeId.energy <= E_thread]
        if not list_request:
            return -1
        time_dead = [self.net.nodes()[index].energy / e[index] for index in list_request]

        near_charge = [self.getNearCharge(self.net.pos[sr]) for index, sr in enumerate(self.net.nodes())]
        distance_charge = [distance(near_charge[i], self.net.pos[self.net.nodes()[i]]) for i, _ in enumerate(self.net.nodes())]
        time_charge = [
            (self.E_sensor_max - self.net.nodes()[index].energy) / (self.alpha / ((distance_charge[index] + self.beta) ** 2) - e[index]) if
            self.alpha / ((distance_charge[index] + self.beta) ** 2) - e[index] > 0 else 10 ** 10 for i, index in enumerate(list_request)]
        time_move = [distance(mc_location, self.net.pos[self.net.nodes()[index]]) / self.velocity for index in
                     list_request]
        time_charge_full = [time_move[i] + time_charge[i] for i, _ in enumerate(list_request)]
        candidate_list = [(index, time_charge_full[i]) for i, index in enumerate(list_request) if
                          time_charge_full[i] <= min(time_dead)]
        min_x = 10 ** 10
        idNode = -1
        if candidate_list:
            for i, item in enumerate(candidate_list):
                if item[1] < min_x:
                    min_x = item[1]
                    idNode = item[0]
        else:
            for i, item in enumerate(time_charge_full):
                if item < min_x:
                    min_x = item
                    idNode = list_request[i]
        d = distance(mc_location, near_charge[idNode])
        # print mc_location, self.net.pos[self.net.nodes()[idNode]], d
        return idNode, d / self.velocity, min_x - d / self.velocity

    # def chargePerSecINMA(self, idNode, t, time_to_charge, timeStop):
    #     p = 0
    #     if 0 < t - (timeStop - time_to_charge) < 1:
    #         p = (self.alpha / self.beta ** 2) * (t - (timeStop - time_to_charge))
    #     elif 0 < timeStop - t < 1:
    #         p = (self.alpha / self.beta ** 2) * (timeStop - t)
    #     elif timeStop - time_to_charge < t < timeStop:
    #         p = self.alpha / self.beta ** 2
    #     self.net.nodes()[idNode].energy = self.net.nodes()[idNode].energy + min(
    #         self.E_sensor_max - self.net.nodes()[idNode].energy, p)

    def chargePerSecINMA(self, mc_location, t, timeStart, time_to_move):
        for index, nd in enumerate(self.net.nodes()):
            p = 0
            d = distance(self.net.pos[nd], mc_location)
            if 0 < t - (timeStart + time_to_move) < 1:
                p = (self.alpha / (d + self.beta) ** 2) * (t - (timeStart + time_to_move))
            elif 1 < t - (timeStart + time_to_move):
                p = (self.alpha / (d + self.beta) ** 2)
            nd.energy = nd.energy + min(self.E_sensor_max - nd.energy, p)
            self.E_mc = self.E_mc - min(self.E_sensor_max - nd.energy, p)
            # print mc_location

    def getNearCharge(self, sr):
        d_min = 10**10
        near_loc = (0, 0)
        for loc_charge in self.charge_pos:
            d = distance(loc_charge, sr)
            if d < d_min:
                d_min = d
                near_loc = loc_charge
        return near_loc

    def simINMA(self, file_name="log_file.csv", num_point=100, thread=0.8):
        t = 1
        isFree = True  # isFree = true if MC is not working
        timeStart = 0.0
        timeStop = 0.0
        list_energy = []
        temp_e = [0.0 for _ in range(self.numNode)]
        mc_location = (0.0, 0.0)

        # calculate time to move
        charge_pos_extend = [(0, 0)]
        charge_pos_extend.extend(self.charge_pos)
        charge_pos_extend.extend([(0, 0)])
        time_move = [distance(charge_pos_extend[i], charge_pos_extend[i + 1]) / self.velocity for i in range(len(charge_pos_extend) - 1)]
        E_move = sum(time_move) * self.e_move
        while t <= self.MaxTime:
            # get log energy to estimate e and E
            if t % 100 == 0:
                print t, min([node.energy for node in self.net.nodes()]), max([node.energy for node in self.net.nodes()])
            # temp_e1 is used to save the used energy each second
            temp_e1 = self.communicateUniform(t, self.delta, freq=self.freq)
            if temp_e1 == -1:
                break
            else:
                temp_e = [temp_e[index] + temp_e1[index] for index in range(self.numNode)]
            # if t % delta = 0, network will log used energy information to the base.
            if t % self.delta == 0:
                temp_list_energy = [(temp_e[index] / self.delta, t) for index in range(self.numNode)]
                temp_e = [0.0 for _ in range(self.numNode)]
                list_energy.append(temp_list_energy)

            # if t < delta, mc does not have energy information to estimate
            if t < self.delta:
                t = t + 1
                continue

            # calculate charging time
            if t == math.ceil(timeStop):
                print t, "node id ", idNode, self.net.nodes()[idNode].energy
                isFree = True

            if isFree:  # isFree = true: MC is free, calculate the next charging position
                E, e = self.estimateEnergy(list_energy=list_energy, num_point=num_point)
                check = self.getNodeDead(e, thread, mc_location)
                if check == -1:
                    t = t + 1
                    continue
                idNode, time_to_move, time_to_charge = check
                print idNode, time_to_move, time_to_charge
                mc_location = self.getNearCharge(self.net.pos[self.net.nodes()[idNode]])
                isFree = False
                timeStart = t + 1
                # timeStop = timeStart + time_to_move + time_to_charge

            if not isFree:  # flag = true: MC in charging time
                self.chargePerSecINMA(mc_location, t, timeStart=timeStart, time_to_move=time_to_move)
                # print idNode, self.net.nodes()[idNode].energy
            if self.net.nodes()[idNode].energy >= self.E_sensor_max - 10**-3:
                isFree = True
            if self.E_mc < E_move:
                d = distance(mc_location, (0.0, 0.0))
                self.E_mc = self.E_mc - d / self.velocity * self.e_move
                time = (self.E_mc_max - self.E_mc) / self.e_mc + d / self.velocity
                self.E_mc = self.E_mc_max
                mc_location = (0.0, 0.0)
                timeStop = t + time

            t = t + 1
