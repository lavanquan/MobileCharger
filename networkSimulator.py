from pymote import *
import pandas as pd
from ast import literal_eval
import math
import csv
from estimateTime import EstimateTime

"""
@author: quanlv
@since: June 13, 2019
@version: 1.0
"""
class NetworkSimulator:
    # doc cac tham so tu file data, index de chi bo du lieu, trong file data - moi dong la 1 bo du lieu
    def __init__(self, file_name="data.csv", index=0):
        df = pd.read_csv(file_name)

        # cac tham so cua sensor
        self.node_pos = list(literal_eval(df.node_pos[index]))  # vi tri cac node
        self.numNode = len(self.node_pos)  # so luong node
        self.energy = df.energy[index]  # nang luong max
        self.commRange = df.commRange[index]  # ban kinh truyen
        self.prob = [int(item) for item in df.prob[index].split(",")]  # tan xuat sinh goi tin cua cac node

        # cac tham so cua mobile charger
        self.charge_pos = list(literal_eval(df.charge_pos[index]))  # vi tri sac
        self.numCharge = len(self.charge_pos) # so luong vi tri mobile charger dung chan
        self.chargeRange = df.chargeRange[index] # ban kinh sac
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

        # network
        self.net = Network()
        idx = 0
        for node in self.node_pos:
            self.net.add_node(pos=node, commRange=self.commRange, energy=self.energy, prob=self.prob[idx])
            idx = idx + 1

    # Tinh khoang cach giua hai toa do
    def distance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

    # Xac dinh node se nhan goi tin quang ba
    def destinate(self, node):
        minDis = self.distance(self.net.pos[node], self.base)
        idNode = -1
        if self.distance(self.net.pos[node], self.base) <= self.commRange:
            idNode = self.base
        else:
            for nd in self.net.neighbors(node):
                d = self.distance(self.net.pos[nd], self.base)
                if d < minDis:
                    minDis = d
                    idNode = nd
        return idNode

    # Xac dinh duong di cua goi tin tu node den base
    def rounte(self, node):
        path = [node]
        if self.distance(self.net.pos[node], self.base) <= self.commRange:
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
            d = self.distance(self.net.pos[send], self.base)
        else:
            d = self.distance(self.net.pos[send], self.net.pos[receive])
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
            return  True
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
            row={}
            for node in self.net.nodes():
                if self.isDead(node) ==  True:
                    row[str(node.id)]=0
                else:
                    row[str(node.id)]=node.energy
            writer.writerow(row)
        csv_file.close()

    # Thuc hien truyen tin trong mang tai thoi diem t voi tham so gui goi tin nang luong delta
    # Sau nay chi can goi ham communicate() thi toan mang se duoc thuc hien gui tin
    def communicate(self, t, delta):
        for node in self.net.nodes():
            if t % node.prob == 0:
                self.send(node, self.b)
        if t % delta == 0:
            for node in self.net.nodes():
                self.send(node, self.b_energy)

    # nang luong sac moi giay cua mobile charger cho toan mang
    # t: thoi diem hien tai
    # timeStart: thoi diem bat dau hanh trinh cua mobile charger
    # chare_time: danh sach thoi gian dung chan cua mobile charger
    # time_move: thoi gian mobile charger di chuyen qua cac diem dung chan
    def chargePerSec(self, t, timeStart, charge_time, time_move):
        index = -1
        for i in range(len(charge_time)):
            if t > timeStart + sum(time_move[:i+1]) + sum(charge_time[:i]) \
                    and t <= timeStart + sum(time_move[:i+1]) +sum(charge_time[:i+1]):
                index = i
        if index != -1:
            start = timeStart + sum(time_move[:index + 1]) + sum(charge_time[:index])
            end = start + charge_time[index]
            if t - start < 1:
                for node in self.net.nodes():
                    d = self.distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        node.energy += self.alpha / ((d + self.beta) ** 2) * (t - start)
            elif end - t < 1:
                for node in self.net.nodes():
                    d = self.distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        node.energy += self.alpha / ((d + self.beta) ** 2) * (end - t)
            else:
                for node in self.net.nodes():
                    d = self.distance(self.net.pos[node], self.charge_pos[index])
                    if d <= self.chargeRange:
                        node.energy += self.alpha / ((d + self.beta) ** 2)


    # ham thuc hien mo phong qua trinh mang thuc thi
    # file_name la ten file se log du lieu nang luong cua mang
    def sim(self, file_name="log_file.csv"):
        t = 1
        self.writeHeader(file_name)

        # timeStart: thoi gian bat dau sac
        # timeStop: thoi gian ket thuc sac
        timeStart = self.delta
        timeStop = self.delta
        flag = False

        dis_charge_pos = sum([self.distance(self.charge_pos[i], self.charge_pos[i+1]) \
                              for i in range(len(self.charge_pos)-1)])
        charge_pos_extend = [(0,0)]
        charge_pos_extend.extend(self.charge_pos)
        charge_pos_extend.extend([(0,0)])
        time_move = [self.distance(charge_pos_extend[i],charge_pos_extend[i+1])/self.velocity \
                     for i in range(len(charge_pos_extend)-1)]
        print time_move
        while t <= self.MaxTime:
            if t == self.delta:
                E = []
                e = []
                for node in self.net.nodes():
                    E.append(node.energy)
                    e.append(0.5)
                estimate = EstimateTime(E=E, e=e, node_pos=self.node_pos, charge_pos=self.charge_pos, \
                                              time_move=time_move, chargeRange=self.chargeRange, \
                                              alpha=self.alpha, beta=self.beta)
                charge_time, T = estimate.calculate()

                timeStart = timeStop + T
                timeStop = timeStart + sum(time_move) + sum(charge_time)


            if t == math.ceil(timeStart):
                flag = True
            elif t == math.ceil(timeStop):
                flag = False
                estimate = EstimateTime(E=E, e=e, node_pos=self.node_pos, charge_pos=self.charge_pos, \
                                              time_move=time_move, chargeRange=self.chargeRange,\
                                              alpha=self.alpha, beta=self.beta)
                charge_time, T = estimate.calculate()
                timeStart = timeStop + T
                timeStop = timeStart + sum(time_move) + sum(charge_time)


            if flag == True:
                self.chargePerSec(t, timeStart, charge_time, time_move)

            self.communicate(t, self.delta)
            if t % self.delta == 0:
                self.write(file_name)

            t += 1