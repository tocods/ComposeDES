name = consumer
time_delta = 1ns
broker = tcp://localhost:5583
values
    wake
        topic = orchestrator/wake
        default = ""
        type = string
        list = false
