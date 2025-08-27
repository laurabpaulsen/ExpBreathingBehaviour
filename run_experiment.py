"""
VERSION B - discriminating weak index and ring finger targets following three salient rhythm-establishing stimuli presented to both fingers
"""
from operator import index
import time
from pathlib import Path
from typing import Union,List, Tuple, Optional
from collections import Counter
import random
# local imports
from utils.experiment import Experiment
from utils.SGC_connector import SGCConnector, SGCFakeConnector

N_REPEATS_BLOCKS = 4
N_SEQUENCE_BLOCKS = 6
RESET_QUEST = 8 # reset QUEST every x blocks
ISIS = [1.33, 1.41, 1.58, 1.82, 2.02]




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
            ISI_adjustment_factor: float = 0.1,
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
            ISI_adjustment_factor = ISI_adjustment_factor,
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
        }

    return trigger_mapping


if __name__ == "__main__":

    start_intensities = {"salient": 3.0, "weak": 1.} # SALIENT NEEDS TO BE AT LEAST xx BIGGER THAN WEAK

    connectors = {
        "middle":  SGCConnector(port="/dev/tty.usbserial-A50027ER", intensity_codes_path=Path("intensity_code.csv"), start_intensity=1),
        "index": SGCConnector(port="/dev/tty.usbserial-A50027EN", intensity_codes_path=Path("intensity_code.csv"), start_intensity=1),
        #"right": SGCFakeConnector(intensity_codes_path=Path("intensity_code.csv"), start_intensity=1)
    }

    # wait 2 seconds
    time.sleep(2)

    for side, connector in connectors.items():
        connector.set_pulse_duration(100)
        connector.change_intensity(start_intensities["salient"])

    block_types = [0, 1, 2, 3, 4]
    wanted_transitions = [(a, b) for a in block_types for b in block_types if a != b]
    order = []

    starting_block = block_types.copy()

    for i in range(N_REPEATS_BLOCKS):
        start_block = random.choice(starting_block)
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
        ISIs=ISIS,#[1.5, 2, 2.5],
        trigger_mapping=create_trigger_mapping(),
        send_trigger=True,
        logfile = Path("output/test_SGC.csv"),
        SGC_connectors=connectors,
        prop_middle_index=[1/2, 1/2],
        break_sound_path=None#Path("/Users/au661930/Library/CloudStorage/OneDrive-Aarhusuniversitet/Dokumenter/projects/_BehaviouralBreathing/code/ExpBreathingBehaviour/utils/sound.wav")
    )
    
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
    block_counts = Counter(order)

    # Count transitions
    transitions = list(zip(order, order[1:]))
    transition_counts = Counter(transitions)
    # Display results
    print("Block counts:")
    for block, count in block_counts.items():
        print(f"  {block}: {count}")

    print("\nTransition counts:")
    for (a, b), count in transition_counts.items():
        print(f"  ({a} → {b}): {count}")


    experiment.run()



