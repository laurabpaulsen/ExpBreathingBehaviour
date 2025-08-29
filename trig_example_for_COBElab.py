from utils.triggers import create_trigger_mapping


from psychopy import parallel
from psychopy.clock import CountdownTimer


port = parallel.ParallelPort(address=0x3FD8)
trigger_mapping = create_trigger_mapping()
countdown_timer = CountdownTimer()


def setParallelData(code=1):
    port.setData(code)
    

def raise_and_lower_trigger(trigger, duration=0.001):
    setParallelData(trigger)

    countdown_timer.reset(duration)
    while countdown_timer.getTime() > 0:
        pass
    setParallelData(0)


for i in range(1, 6):
    for description, trig_val in trigger_mapping.items():
        print(description, trig_val)
        countdown_timer.reset(0.5)
        while not countdown_timer.getTime() > 0:
            pass
        raise_and_lower_trigger(trigger=trig_val)