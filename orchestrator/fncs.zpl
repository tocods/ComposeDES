name = orchestrator
time_delta = 1ns
broker = tcp://localhost:5570
values
    compute/completed
        topic = gpusim/compute/completed
        default = ""
        type = string
        list = false
    network/completed
        topic = ns3/network/completed
        default = ""
        type = string
        list = false
