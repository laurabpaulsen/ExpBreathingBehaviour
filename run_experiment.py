"""
VERSION B - discriminating weak index and ring finger targets following three salient rhythm-establishing stimuli presented to both fingers


To run on stim PC at COBE lab run `python3.8 run_experiment.py` from the command prompt

"""
import time
from pathlib import Path
from typing import Union,List, Tuple, Optional
from collections import Counter
import random
import os
import numpy as np

# local imports
from utils.experiment import Experiment
from utils.SGC_connector import SGCConnector, SGCFakeConnector
from utils.triggers import create_trigger_mapping


# CONFIG
# -------------------
N_REPEATS_BLOCKS = 4
N_SEQUENCE_BLOCKS = 9
RESET_QUEST = 2 # how many blocks before resetting QUEST
ISIS = [1.29, 1.44, 1.57, 1.72] # !ISIS = [1.33, 1.41, 1.58, 1.82, 2.02]not a problem with these ISIs, could potentially add one a bit shorter! 
VALID_INTENSITIES = np.arange(1.0, 10.1, 0.1).round(1).tolist()

OUTPUT_PATH = Path(__file__).parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

# check whether it is running on mac or windows
if os.name == "posix":
    # macOS
    index_connector_port = "/dev/tty.usbserial-A50027EN"
    middle_connector_port = "/dev/tty.usbserial-A50027ER"
else:
    # Windows
    index_connector_port = "COM6"
    middle_connector_port = "COM7"


# UTILITIES
# -------------
def build_block_order(
    wanted_transitions: List[Tuple[int, int]],
    start_blocks: Optional[List[int]] = None
) -> List[int]:
    """
    Build a block order that exactly produces the given list of transitions.

    Parameters:
        wanted_transitions (list of tuples): Each tuple represents a transition (e.g., (0, 1)).
        start_blocks (list of int, optional): Block types to consider as starting points. 
                                              Defaults to all blocks present in wanted_transitions.

    Returns:
        list of int: A sequence of blocks that yields the specified transitions.

    Raises:
        ValueError: If no valid order can be found.
    """
    wanted_counter = Counter(wanted_transitions)

    # Infer block types from transition tuples
    block_types = list(set(b for pair in wanted_transitions for b in pair))

    def backtrack(path):
        if sum(wanted_counter.values()) == 0:
            return path

        last = path[-1]
        next_options = block_types[:]
        random.shuffle(next_options)

        for next_block in next_options:
            if next_block == last:
                continue
            candidate = (last, next_block)
            if wanted_counter[candidate] > 0:
                wanted_counter[candidate] -= 1
                result = backtrack(path + [next_block])
                if result:
                    return result
                wanted_counter[candidate] += 1  # backtrack

        return None

    # If not specified, start from any available block type
    if start_blocks is None:
        start_blocks = block_types
    else:
        start_blocks = [s for s in start_blocks if s in block_types]

    for start_block in random.sample(start_blocks, len(start_blocks)):
        result = backtrack([start_block])
        if result:
            return result

    raise ValueError("No valid order found")



def print_experiment_information(experiment):

    duration = experiment.estimate_duration()
    print(f"The experiment is estimated to last {duration/60} minutes")
    experiment.setup_experiment()

    # Extract event_type from each dictionary
    event_types = [event['event_type'] for event in experiment.events if not event == "break"]

    # Count each event_type
    event_counts = Counter(event_types)

    # Print the results
    for event_type, count in event_counts.items():
        print(f"{event_type}: {count}")

    # Count individual blocks
    block_counts = Counter(experiment.order)

    # Count transitions
    transitions = list(zip(experiment.order, experiment.order[1:]))
    transition_counts = Counter(transitions)
    # Display results
    print("Block counts:")
    for block, count in block_counts.items():
        print(f"  {block}: {count}")

    print("\nTransition counts:")
    for (a, b), count in transition_counts.items():
        print(f"  ({a} → {b}): {count}")

def get_participant_info():
    pid = input("Enter participant ID: ").strip()

    while True:
        try:
            salient = float(input("Enter salient intensity (1.0–10.0): "))
            if salient not in VALID_INTENSITIES:
                raise ValueError
            break
        except ValueError:
            print("❌ Invalid input. Please enter a number between 1.0 and 10.0 in steps of 0.1.")

    weak = np.round(salient / 2, 1)
    if weak not in VALID_INTENSITIES:
        raise ValueError(f"Weak intensity {weak} is invalid. Adjust salient value.")

    print("\nParticipant Information:")
    print(f" ID: {pid}")
    print(f" Salient Intensity: {salient}")
    print(f" Weak Intensity: {weak}")


    confirm = input("Is this information correct? (y/n): ").strip().lower()
    if confirm != "y":
        print("Exiting experiment setup.")
        exit()

    return pid, {"salient": salient, "weak": weak}


class MiddleIndexTactileDiscriminationTask(Experiment):
    def __init__(
            self, 
            trigger_mapping: dict,
            ISIs: List[float],
            order: List[int],
            n_sequences: int = 10,
            prop_middle_index: List[float] = [0.5, 0.5],
            intensities: dict = {"salient": 6.0, "weak": 2.0},
            trigger_duration: float = 0.001,
            QUEST_target: float = 0.75,
            reset_QUEST: Union[int, bool] = False, # how many blocks before resetting QUEST
            QUEST_plus: bool = True,
            send_trigger: bool = False,
            logfile: Path = Path("data.csv"),
            SGC_connectors = None,
            break_sound_path=None
            ):
        
        super().__init__(
            trigger_mapping = trigger_mapping,
            ISIs = ISIs,
            order = order,
            n_sequences = n_sequences,
            prop_target1_target2 = prop_middle_index,
            target_1="middle",
            target_2="index",
            intensities = intensities,
            trigger_duration = trigger_duration,
            QUEST_target = QUEST_target,
            reset_QUEST = reset_QUEST,
            QUEST_plus = QUEST_plus,
            send_trigger = send_trigger,
            logfile = logfile,
            break_sound_path = break_sound_path
        )
        self.SGC_connectors = SGC_connectors
    
    def deliver_stimulus(self, event_type):
        if self.SGC_connectors:
            if "salient" in event_type:  # send to both fingers
                for connector in self.SGC_connectors.values():
                    connector.send_pulse()
            elif self.SGC_connectors and "target" in event_type: # send to the finger specified in the event type
                self.SGC_connectors[event_type.split("/")[-1]].send_pulse()

    def prepare_for_next_stimulus(self, event_type, next_event_type):
        if self.SGC_connectors:
            # after sending the trigger for the weak target stimulation change the intensity to the salient intensity
            if "target" in event_type: 
                self.SGC_connectors[event_type.split("/")[-1]].change_intensity(self.intensities["salient"])

            # check if next stimuli is weak, then lower based on which!
            if "target" in next_event_type:
                self.SGC_connectors[next_event_type.split("/")[-1]].change_intensity(self.intensities["weak"])

    def trial_block(self, ISI=1.5, n_sequences=N_SEQUENCE_BLOCKS):
        trial_sequence_events = self.event_sequence(n_sequences=n_sequences, ISI=ISI, block_idx="trial", reset_QUEST=None)
        
        self.response_handler.start()  # Start the keyboard listener
        self.response_handler.enable()  # Enable response handling
        self.loop_over_events(trial_sequence_events, log_file=None)
        self.response_handler.stop()  # Stop the keyboard listener



if __name__ == "__main__":
    # --- Collect participant info ---
    participant_id, start_intensities = get_participant_info()

    # Setup logfile based on participant ID
    logfile = OUTPUT_PATH / f"{participant_id}_behavioural_data.csv"

    # check if it already exists
    if logfile.exists():
        i = 1
        while logfile.exists():
            logfile = OUTPUT_PATH / f"{participant_id}_behavioural_data_{i}.csv"
            i += 1

    print(f"Behavioural data will be saved to: {logfile}")

    connectors = {
        #"middle":  SGCConnector(port=middle_connector_port, intensity_codes_path=Path("intensity_code.csv"), start_intensity=1),
        #"index": SGCConnector(port=index_connector_port, intensity_codes_path=Path("intensity_code.csv"), start_intensity=1),
        "middle": SGCFakeConnector(intensity_codes_path=Path("intensity_code.csv"), start_intensity=1),
        "index": SGCFakeConnector(intensity_codes_path=Path("intensity_code.csv"), start_intensity=1)
    }

    # wait 2 seconds
    time.sleep(2)

    for side, connector in connectors.items():
        connector.set_pulse_duration(100)
        connector.change_intensity(start_intensities["salient"])

    block_types = list(range(len(ISIS)))  # one block type per ISI
    wanted_transitions = [(a, b) for a in block_types for b in block_types if a != b]
    order = []
    
    starting_block = block_types.copy()

    for i in range(N_REPEATS_BLOCKS):
        try:
            start_block = random.choice(starting_block)
        except IndexError:
            starting_block = block_types.copy()
        starting_block.remove(start_block)

        tmp_order = build_block_order(
            wanted_transitions, 
            start_blocks=[start_block])
        order.extend(tmp_order)
        order.append("break")

    
    experiment = MiddleIndexTactileDiscriminationTask(
        intensities=start_intensities,
        n_sequences=N_SEQUENCE_BLOCKS,
        order = order,
        QUEST_plus=False,
        reset_QUEST=RESET_QUEST, # reset QUEST every x blocks
        ISIs=ISIS,
        trigger_mapping=create_trigger_mapping(),
        trigger_duration=0.005,
        send_trigger=False, # after running the trial block it is set to True
        logfile = logfile,
        SGC_connectors=connectors,
        prop_middle_index=[1/2, 1/2],
        break_sound_path=None #Path("/Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/ExpBreathingBehaviour/utils/sound.wav")
    )


    start_time = time.time()
    print_experiment_information(experiment)
    experiment.check_in_on_participant(message="Ready to begin practice block.")
    #experiment.trial_block(ISI=1.5, n_sequences=10) # practice block
    experiment.send_trigger = True
    experiment.check_in_on_participant(message="Ready to begin main experiment.")
    experiment.run()

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Experiment completed in {total_duration/60:.2f} minutes.")



