"""
Description: This file contains the code for sending triggers to the neuroimaging system.
"""
# -*- coding: utf-8 -*-
from psychopy import parallel


port = parallel.ParallelPort(address=0x3FD8)
print(f"Parallel port {port} initialised.")

# Figure out whether to flip pins or fake it
try:
    port.setData(1)
except NotImplementedError:
    print(f"Parallel port {port} not implemented???.")
    def setParallelData(code=1):
        if code > 0:
            # logging.exp('TRIG %d (Fake)' % code)
            print('TRIG %d (Fake)' % code)
            pass
else:
    port.setData(0)
    setParallelData = port.setData


def create_trigger_mapping(
        stim = 1,
        target = 2,
        middle = 4,
        index = 8,
        response = 16,
        correct = 32,
        incorrect = 64):
    
    trigger_mapping = {
        "stim/salient": stim,
        "target/middle": target + middle,
        "target/index": target + index,
        "response/index/correct": response + index + correct,
        "response/middle/incorrect": response + middle + incorrect,
        "response/middle/correct": response + middle + correct,
        "response/index/incorrect": response + index + incorrect,
        "break/start": 128,
        "break/end": 129,
        "experiment/start": 254,
        "experiment/end": 255
        }

    return trigger_mapping

