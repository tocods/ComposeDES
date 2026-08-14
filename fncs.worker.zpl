name = ns3
time_delta = 1ns
broker = tcp://localhost:5570
values
    network/dispatch
        topic = orchestrator/network/dispatch
        default = ""
        type = string
        list = false
    control
        topic = orchestrator/control
        default = ""
        type = string
        list = false
