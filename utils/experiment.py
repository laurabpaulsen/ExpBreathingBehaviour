from pathlib import Path
import numpy as np
from numpy.random import choice
from typing import Union, List
import time

from psychopy.clock import CountdownTimer
from psychopy.data import QuestPlusHandler, QuestHandler

from utils.responses import KeyboardListener


class Experiment:
    LOG_HEADER = "time,block,ISI,intensity,event_type,trigger,n_in_block,correct,QUEST_reset\n"

    def __init__(
            self, 
            trigger_mapping: dict,
            ISIs: List[float],
            order: List[int] = None,
            n_sequences: int = 10, 
            prop_target1_target2: List[float] = [0.5, 0.5], 
            target_1: str = "left",
            target_2: str = "right",
            intensities: dict = {"salient": 6.0, "weak": 2.0},
            trigger_duration: float = 0.001,
            send_trigger: bool = False,
            QUEST_target: float = 0.75,
            reset_QUEST: Union[int, bool] = False, # how many blocks before resetting QUEST
            QUEST_plus: bool = True,
            ISI_adjustment_factor: float = 0.1,
            logfile: Path = Path("data.csv"),
            break_sound_path: Union[Path, str, None] = None,
            ):
        """
        Initializes the parameters and attributes for the experimental paradigm.

        Parameters
        ----------
    
        order : list, optional
            The sequence order of stimuli types in the experiment, represented as 
            indices (0, 1, 2, etc.). Defaults to [0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 0, 1].
        
        n_sequences : int, optional
            Number of sequences in each block. Defaults to 10.
        
        prop_target1_target2 : list, optional
            Proportions of target1 and target2.
            Defaults to [0.5, 0.5].
        
        intensities : dict, optional
            A dictionary mapping stimulus types ("salient", "weak") to intensity values.
            Defaults to {"salient": 4.0, "weak": 2.0}.
        
        trigger_mapping : dict, optional
            A mapping of stimulus and response types to specific trigger values sent
            during the experiment.
        
        QUEST_target : float, optional
            Target proportion of correct responses for QUEST to adjust intensity.
            Defaults to 0.75.
        
        trigger_duration : float, optional
            Duration of the trigger signal, in seconds. Defaults to 0.001.
        
        reset_QUEST : int or bool, optional
            Determines if and how frequently the QUEST algorithm should reset.
            Set to an integer for resets every X blocks or False to disable resetting.
            Defaults to False.
        
        ISI_adjustment_factor : float, optional
            Factor for adjusting inter-stimulus intervals dynamically based on respiratory rate. Defaults to 0.1.
        
        logfile : Path, optional
            Path to the log file for saving experimental data. Defaults to Path("data.csv").

        Returns
        -------
        None
        """
        
        self.ISIs = ISIs
        self.reset_QUEST = reset_QUEST
        self.logfile = logfile
        self.n_sequences = n_sequences
        self.order = order
        self.trigger_mapping = trigger_mapping
        self.prop_target1_target2 = prop_target1_target2
        self.trigger_duration = trigger_duration
        self.send_trigger = send_trigger
        if self.send_trigger:
            from utils.triggers import setParallelData
        self.countdown_timer = CountdownTimer() 
        self.events = []

        self.ISI_adjustment_factor = ISI_adjustment_factor
        
        self.target_1 = target_1
        self.target_2 = target_2

        # for response handling 
        self.listener = KeyboardListener()
        self.keys_target = {
            target_1: ['2', 'y'],
            target_2: ['1', 'b']
        }
        
        # QUEST parameters
        self.intensities =  intensities 
        self.QUEST_start_val = intensities["weak"] # NOTE: do we want to reset QUEST with the startvalue or start from a percentage of the weak intensity stimulation?
        self.max_intensity_weak = intensities["salient"] - 0.5
        self.QUEST_plus = QUEST_plus
        self.QUEST_target = QUEST_target 
        self.QUEST_reset()
        self.break_sound_path = break_sound_path
        if self.break_sound_path:
            from playsound import playsound

    def play_break_sound(self):
        # Play a sound to indicate a break
        if self.break_sound_path:
            playsound(str(self.break_sound_path))

    def setup_experiment(self):
        for block_idx, block in enumerate(self.order):
            if block == "break":
                self.events.append("break")
            else:
                ISI = self.ISIs[block]

                # check if QUEST needs to be reset in this block
                if self.reset_QUEST and block_idx % self.reset_QUEST == 0 and block_idx != 0:
                    reset = int( self.n_sequences/2) # approximately halfway through the block
                else:
                    reset = False
            
                self.events.extend(self.event_sequence(self.n_sequences, ISI, block_idx, reset_QUEST=reset))
        
    def event_sequence(self, n_sequences, ISI, block_idx, n_salient=3, reset_QUEST: Union[int, None] = None) -> List[dict]:
        """
        Generate a sequence of events for a block

        reset_QUEST: int or None
            If an integer, the QUEST procedure will be reset after this many sequences
        """
        event_counter_in_block = 0

        events = []
        for seq in range(n_sequences):
            
            # checking if it is time for a QUEST reset
            reset = reset_QUEST and seq == reset_QUEST

            for i in range(n_salient):
                event_counter_in_block += 1
                events.append({"ISI": ISI, "event_type": "stim/salient", "n_in_block": event_counter_in_block, "block": block_idx, "reset_QUEST": reset})
                if reset:
                    reset=False
            
            event_counter_in_block += 1
            event_type = choice([self.target_1, self.target_2], 1, p=self.prop_target1_target2)
            

            events.append({"ISI": ISI, "event_type": f"target/{event_type[0]}", "n_in_block": event_counter_in_block, "block": block_idx, "reset_QUEST": False})

        return events
    
    def make_QUEST_handler(self):
        if self.QUEST_plus:
            return QuestPlusHandler(
                startIntensity=self.QUEST_start_val,
                intensityVals=[round(i, 1) for i in np.arange(1.0, self.max_intensity_weak, 0.1)],
                thresholdVals=[round(i, 1) for i in np.arange(1.0, self.max_intensity_weak, 0.1)],
                stimScale="linear",
                responseVals=(1, 0),
                slopeVals=[3, 4, 5],
                lowerAsymptoteVals=0.5,
                lapseRateVals=0.05,
                nTrials=None
            )
        else:
            return QuestHandler(
                startVal=self.QUEST_start_val,
                startValSd=1.0,
                minVal=1.0,
                maxVal=self.max_intensity_weak,
                pThreshold=self.QUEST_target,
                stepType="linear",
                nTrials=None,
                beta=3.5,
                gamma=0.5,
                delta=0.01
            )
        
    
    def QUEST_reset(self):
        """Reset the QUEST procedure and update intensity."""
        self.QUEST = self.make_QUEST_handler()
        self.update_weak_intensity()
        print("QUEST has been reset")

    def update_weak_intensity(self):
        """
        Update the weak intensity based on the QUEST procedure!
        """
        proposed_intensity = self.QUEST.next()
        self.intensities["weak"] = round(proposed_intensity, 1)

    def deliver_stimulus(self, event_type):
        pass
    
    def prepare_for_next_stimulus(self, event_type, next_event_type):
        pass

    def check_in_on_participant(self):
        input("Check in on the participant. Press Enter to continue...")
        time.sleep(1)


    def loop_over_events(self, events: List[dict], log_file):
        """
        Loop over the events in the experiment
        """
        for i, trial in enumerate(events):
            if trial == "break":
                # play wav sound to alert participant and experimenter
                self.play_break_sound()
                self.check_in_on_participant()
                continue
            
            event_type = trial["event_type"]

            intensity = self.intensities["salient"] if "salient" in event_type else self.intensities["weak"]

            if self.send_trigger:
                self.raise_and_lower_trigger(self.trigger_mapping[event_type])  # Send trigger
            # deliver pulse
            self.deliver_stimulus(event_type)
            
            event_time = time.perf_counter() - self.start_time
            
            self.log_event(
                **trial,
                event_time = event_time,
                intensity=intensity,
                trigger=self.trigger_mapping[event_type], 
                correct="NA",
                log_file = log_file
                )
            
            print(f"Event: {event_type}, intensity: {intensity}")

            # Check if this is a target event
            self.listener.active = "target" in event_type

            target_time = event_time + trial["ISI"] + self.start_time
            response_given = False # to keep track of whether a response has been given

            try: 
                self.prepare_for_next_stimulus(event_type, events[i+1]["event_type"])
            except IndexError:
                pass
            except TypeError: # if break is coming up next
                try:
                    self.prepare_for_next_stimulus(event_type, events[i+2]["event_type"])
                except IndexError:
                    pass

            if trial["reset_QUEST"]:
                self.QUEST_reset()

            while time.perf_counter() < target_time:
                # check for key press during target window
                if self.listener.active and not response_given:
                    key = self.listener.get_response()
                    if key:
                        correct, response_trigger = self.correct_or_incorrect(key, event_type)
                        print(f"Response: {key}, Correct: {correct}")
                        self.raise_and_lower_trigger(response_trigger) 
                        response_given = True
                        
                        # overwrite event type for logging
                        trial["event_type"] = "response"
                        
                        self.log_event(
                            **trial,
                            event_time=time.perf_counter() - self.start_time, 
                            intensity=intensity, 
                            trigger=response_trigger, 
                            correct=correct, 
                            log_file = log_file
                            )
                        self.listener.active = False
         
                        self.QUEST.addResponse(correct, intensity = intensity)
                        self.update_weak_intensity()

            # stop listening for responses
            self.listener.active = False


    def log_event(self, event_time, block, ISI, intensity, event_type, trigger, n_in_block, correct, reset_QUEST, log_file):
        log_file.write(f"{event_time},{block},{ISI},{intensity},{event_type},{trigger},{n_in_block},{correct}, {reset_QUEST}\n")
    
    
    def validate_ISI(self) -> bool:
        """
        Validate ISI values to ensure they are within a reasonable range and not multiples of 50 Hz.
        """
        # Check for negative ISI
        if any(ISI < 0 for ISI in self.ISIs):
            print("Warning: ISI is negative, please check the input respiratory rate and adjustment factor.")
            return False

        # Check if ISI is close to a multiple of 1/50 (i.e., could align with 50Hz interference)
        for ISI in self.ISIs:
            if abs((ISI * 50) - round(ISI * 50)) < 1e-2:
                print(f"Warning: ISI {ISI:.4f}s is too close to a multiple of 1/50s (i.e., 50 Hz), may cause electrical noise.")
                return False

        print(f"Valid ISIs: {self.ISIs}")
        return True
    
    def get_user_input_respiratory_rate(self):
        while True:
            try:
                respiratory_rate = float(input("Please input the average length of one respiratory cycle: "))
                if respiratory_rate <= 0:
                    print("Invalid input. Please enter a positive value.")
                else: 
                    break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

        return respiratory_rate

    def adjust_ISI(self, ms_per_breath: float) -> bool:
        """
        Adjust ISI for block A and C based on respiratory rate in block B.
        Returns True if ISIs are valid, False otherwise.
        """
        delta_ISI = self.ISI_adjustment_factor * ms_per_breath
    
        self.ISIs[0] = self.ISIs[1] - delta_ISI
        self.ISIs[2] = self.ISIs[1] + delta_ISI

        print(f"ISI for block A: {self.ISIs[0]}, ISI for block C: {self.ISIs[2]} after adjustment based on average length of one respiratory cycle {delta_ISI}")

        return self.validate_ISI()


    def raise_and_lower_trigger(self, trigger):
        setParallelData(trigger)
        self.countdown_timer.reset(self.trigger_duration)
        while self.countdown_timer.getTime() > 0:
            pass
        setParallelData(0)
    
    def correct_or_incorrect(self, key, event_type):
        if key in self.keys_target[event_type.split('/')[-1]]:
            return 1, self.trigger_mapping[f"response/{event_type.split('/')[-1]}/correct"]
        else:
            return 0, self.trigger_mapping[f"response/{event_type.split('/')[-1]}/incorrect"]
        
    def estimate_duration(self) -> float:
        """
        Estimate the total duration of the experiment in seconds, assuming all ISIs have a mean of ISI[1].
        
        Returns:
            float: Estimated duration of the experiment in seconds.
        """
        mean_ISI = np.mean(self.ISIs)
        n_events_per_sequence = 3 + 1  # 3 salient stimuli + 1 target stimulus per sequence
        total_events = n_events_per_sequence * (self.n_sequences * len([block for block in self.order if block != "break"]))
        
        # Total time is events * ISI + trigger durations
        total_duration = (total_events * mean_ISI) 
        
        return total_duration
    
    def run(self):

        # NOTE! WRITE TO LOG FILE IN THE BREAKS?

        self.listener.start_listener()  # Start the keyboard listener
        self.logfile.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

        self.start_time = time.perf_counter()
       
        with open(self.logfile, 'w') as log_file:
            log_file.write(self.LOG_HEADER)
            
            # determine the respiratory rate during block 
            #self.determine_respiratory_rate(log_file)

            # run the experiment
            #self.setup_experiment()
            self.loop_over_events(self.events, log_file)

        self.listener.stop_listener()  # Stop the keyboard listener