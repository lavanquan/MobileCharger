from networkSimulator import NetworkSimulator
for j in range(1):
    i = j + 0
    print "DataSet ", i
    # if i == 0:
    #     continue
    network = NetworkSimulator("data/thaydoitileguitin.csv", i)

    # network.sim("log_file_greedy.csv", gamma=0.5)
    # network.simNoCharge("log_file_nocharge.csv")
    network.simINMA("log_file_inma.csv", thread=0.4)
    print "Done DataSet ", i
