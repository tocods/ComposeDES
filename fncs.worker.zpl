name = gpusim
time_delta = 1ns
broker = tcp://localhost:5570
values
    compute/dispatch
        topic = orchestrator/compute/dispatch
        default = ""
        type = string
        list = false
    control
        topic = orchestrator/control
        default = ""
        type = string
        list = false
