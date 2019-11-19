from networkSimulator import NetworkSimulator
for j in range(1):
    i = j
    print "DataSet ", i
    # if i == 0:
    #     continue
    network = NetworkSimulator("data/thaydoisonode.csv", i)

    # network.sim("log_file.csv")
    # network.simNoCharge("log_file.csv")
    network.simINMA("log_file.csv", thread=0.4)
    print "Done DataSet ", i
