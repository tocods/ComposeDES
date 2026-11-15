"""Lightweight NVML sampling outside the timed path."""
from threading import Event, Thread
from time import perf_counter_ns


class GPU:
    def __init__(self):
        self.stop = Event()
        self.state = None
        self.thread = Thread(target=self.read, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def read(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            card = nv.nvmlDeviceGetHandleByIndex(0)
            while not self.stop.is_set():
                self.state = dict(ns=perf_counter_ns(),
                                  sm_mhz=nv.nvmlDeviceGetClockInfo(card, nv.NVML_CLOCK_SM),
                                  memory_mhz=nv.nvmlDeviceGetClockInfo(card, nv.NVML_CLOCK_MEM),
                                  temperature_c=nv.nvmlDeviceGetTemperature(card, nv.NVML_TEMPERATURE_GPU),
                                  power_mw=nv.nvmlDeviceGetPowerUsage(card),
                                  util=nv.nvmlDeviceGetUtilizationRates(card).gpu)
                self.stop.wait(.25)
        except Exception as error:
            print("GPU telemetry unavailable:", error, flush=True)
