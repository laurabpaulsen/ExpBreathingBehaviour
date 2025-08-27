"""
Description: This file contains the code for sending triggers to the neuroimaging system.
"""
# -*- coding: utf-8 -*-
from psychopy import parallel
import platform

port = parallel.ParallelPort(address=0x3FD8)
print(f"Parallel port {port} initialised.")

# NB problems getting parallel port working under conda env
# from psychopy.parallel._inpout32 import PParallelInpOut32
# port = PParallelInpOut32(address=0xDFF8)  # on MEG stim PC
# parallel.setPortAddress(address='0xDFF8')
# port = parallel

# Figure out whether to flip pins or fake it
try:
    port.setData(1)
except NotImplementedError:
    def setParallelData(code=1):
        if code > 0:
            # logging.exp('TRIG %d (Fake)' % code)
            print('TRIG %d (Fake)' % code)
            pass
else:
    port.setData(0)
    setParallelData = port.setData

