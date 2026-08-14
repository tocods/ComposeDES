name = ns3
time_delta = 1ns
broker = tcp://localhost:5570
values
    cloudsim/transfer
        topic = gpusim/cloudsim/transfer
        default = ""
        type = string
        list = false
    cloudsim/end
        topic = gpusim/cloudsim/end
