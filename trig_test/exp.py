from triggers import setParallelData
import time

if __name__ == "__main__":
    # Example usage of setParallelData
    for code in range(1, 6):
        setParallelData(code)
        # wait
        time.sleep(0.5)  # wait for half a second before sending the next trigger