from pathlib import Path
import numpy as np
from numpy.random import choice
from typing import Union, List
import time

from psychopy.clock import CountdownTimer
from psychopy import core
from psychopy.data import QuestPlusHandler, QuestHandler

from utils.responses import KeyboardListener
from utils.triggers import setParallelData
from winsound import PlaySound, SND_FILENAME
class Experiment:
    LOG_HEADER = "time,block,ISI,intensity,event_type,trigger,n_in_block,correct,QUEST_reset,rt\n"

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
            QUEST_target: float = 0.60,
            reset_QUEST: Union[int, bool] = False, # how many blocks before resetting QUEST
            QUEST_plus: bool = True,
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

        self.countdown_timer = CountdownTimer() 
        self.events = []
        
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
        self.QUEST_n_resets = 0
        self.QUEST_reset()
        self.break_sound_path = break_sound_path



        self.start_time = time.perf_counter()

    def play_break_sound(self):
        # Play a sound to indicate a break
        if self.break_sound_path:
            PlaySound(str(self.break_sound_path), SND_FILENAME)

    def setup_experiment(self):
        logged_block_idx = 0
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

                self.events.extend(self.event_sequence(self.n_sequences, ISI, logged_block_idx, reset_QUEST=reset))
                logged_block_idx += 1
        
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

        # update the quest start val to the previous weak intensity
        if self.QUEST_n_resets > 0:
            self.QUEST_start_val = min(self.QUEST.mean(), self.max_intensity_weak)
            print(f"QUEST start value updated to: {self.QUEST_start_val}")
        self.QUEST = self.make_QUEST_handler()

        self.update_weak_intensity()
        self.QUEST_n_resets += 1
        print("QUEST has been reset")

    def update_weak_intensity(self):
        """
        Update the weak intensity based on the QUEST procedure!
        """
        proposed_intensity = self.QUEST.next()
    
        # make sure the intensity is not higher than the max allowed
        proposed_intensity = max(1.0, min(proposed_intensity, self.max_intensity_weak))

        self.intensities["weak"] = round(proposed_intensity, 1)

    def deliver_stimulus(self, event_type):
        raise NotImplementedError("Subclasses should implement this!")

    def prepare_for_next_stimulus(self, event_type, next_event_type):
        raise NotImplementedError("Subclasses should implement this!")
    
    def check_in_on_participant(self, message: str = "Check in on the participant.", log_file=None):
        if self.send_trigger:
            self.raise_and_lower_trigger(self.trigger_mapping["break/start"])
            # also log the event¨
            if log_file:
                self.log_event(
                    event_time=time.perf_counter() - self.start_time,
                    block="break",
                    ISI="NA",
                    intensity="NA",
                    event_type="break",
                    trigger=self.trigger_mapping["break/start"],
                    n_in_block="NA",
                    correct="NA",
                    reset_QUEST=False,
                    rt="NA",
                    log_file=log_file
                )

        self.play_break_sound()
        input(message + " Press Enter to continue...")
        if self.send_trigger:
            self.raise_and_lower_trigger(self.trigger_mapping["break/end"])
            if log_file:
                self.log_event(
                    event_time=time.perf_counter() - self.start_time,
                    block="break",
                    ISI="NA",
                    intensity="NA",
                    event_type="break/end",
                    trigger=self.trigger_mapping["break/end"],
                    n_in_block="NA",
                    correct="NA",
                    reset_QUEST=False,
                    rt="NA",
                    log_file=log_file
                )
        time.sleep(2)

    def loop_over_events(self, events: List[dict], log_file):
        """
        Loop over the events in the experiment
        """
        for i, trial in enumerate(events):
            if trial == "break":
                self.check_in_on_participant(log_file=log_file)
                continue
            
            event_type = trial["event_type"]
            trigger = self.trigger_mapping[event_type]

            intensity = self.intensities["salient"] if "salient" in event_type else self.intensities["weak"]

            if self.send_trigger:
                self.raise_and_lower_trigger(trigger)  # Send trigger
            
            # deliver pulse
            self.deliver_stimulus(event_type)
            
            stim_time = time.perf_counter() - self.start_time
            
            if log_file:
                self.log_event(
                    **trial,
                    event_time=stim_time,
                    intensity=intensity,
                    rt="NA",
                    trigger=trigger,
                    correct="NA",
                    log_file=log_file
                )
            
            print(f"Event: {event_type}, intensity: {intensity}")

            # Check if this is a target event
            self.listener.active = "target" in event_type

            target_time = stim_time + trial["ISI"]
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

            while (time.perf_counter() - self.start_time) < target_time:
                # check for key press during target window
                if self.listener.active and not response_given:
                    rt = "NA"
                    key = self.listener.get_response()
                    if key:
                        correct, response_trigger = self.correct_or_incorrect(key, event_type)
                        time_of_response = (time.perf_counter() - self.start_time)

                        print(f"Response: {key}, Correct: {correct}")
                        if self.send_trigger:
                            self.raise_and_lower_trigger(response_trigger)

                        rt = time_of_response - stim_time
                        response_given = True
                        
                        # overwrite event type for logging
                        trial["event_type"] = "response"
                        if log_file:
                            self.log_event(
                                **trial,
                                event_time=time_of_response,
                                intensity="NA",
                                trigger=response_trigger,
                                correct=correct,
                                rt=rt,
                                log_file=log_file
                            )
                        self.listener.active = False

                        self.QUEST.addResponse(correct, intensity=intensity)
                        self.update_weak_intensity()

            if ("target" in event_type) and (not response_given):
                print("No response given")
                # Update QUEST with the guessed outcome and advance intensity
                self.QUEST.addResponse(np.random.choice([0, 1]), intensity=intensity)
                self.update_weak_intensity()

            # stop listening for responses
            self.listener.active = False


    def log_event(self, event_time, block, ISI, intensity, event_type, trigger, n_in_block, correct, reset_QUEST, rt, log_file):
        log_file.write(f"{event_time},{block},{ISI},{intensity},{event_type},{trigger},{n_in_block},{correct},{reset_QUEST},{rt}\n")
    
    
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


    def raise_and_lower_trigger_old(self, trigger):
        setParallelData(trigger)
        self.countdown_timer.reset(self.trigger_duration)
        while self.countdown_timer.getTime() > 0:
            pass
        setParallelData(0)

    def raise_and_lower_trigger(self, trigger):
        setParallelData(trigger)
        core.wait(self.trigger_duration, hogCPUperiod=min(self.trigger_duration, 0.01))
        setParallelData(0)
    
    def correct_or_incorrect(self, key, event_type):
        if key in self.keys_target[event_type.split('/')[-1]]:
            return 1, self.trigger_mapping[f"response/{event_type.split('/')[-1]}/correct"]
        else:
            return 0, self.trigger_mapping[f"response/{event_type.split('/')[-1]}/incorrect"]


    def estimate_duration(self, break_duration: float = 30.0) -> float:
        """
        Estimate the total duration of the experiment in seconds.
        This calculation is based on the actual order of blocks (including breaks),
        the number of sequences per block, and the ISI structure.

        Parameters
        ----------
        break_duration : float
            Estimated duration of breaks in seconds.

        Returns
        -------
        float
            Estimated duration of the experiment in seconds.
        """
        total_duration = 0.0
        
        for block in self.order:
            if block == "break":
                # give an arbitrary pause duration for breaks (e.g. 60 s)
                total_duration += break_duration
                continue

            ISI = self.ISIs[block]
            n_events_per_sequence = 3 + 1  # 3 salient + 1 target per sequence
            n_events = n_events_per_sequence * self.n_sequences
            
            total_duration += n_events * ISI

        return total_duration
    

    def run(self, write_header: bool = True):
        self.listener.start_listener()  # Start the keyboard listener
        self.logfile.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
       
        with open(self.logfile, 'w') as log_file:

            if self.send_trigger:
                self.raise_and_lower_trigger(self.trigger_mapping["experiment/start"])
                if log_file:
                    self.log_event(
                        event_time=time.perf_counter() - self.start_time,
                        block="experiment",
                        ISI="NA",
                        intensity="NA",
                        event_type="experiment/start",
                        trigger=self.trigger_mapping["experiment/start"],
                        n_in_block="NA",
                        correct="NA",
                        reset_QUEST=False,
                        rt="NA",
                        log_file=log_file
                    )


            if write_header:
                log_file.write(self.LOG_HEADER)
            self.loop_over_events(self.events, log_file)

            if self.send_trigger:
                self.raise_and_lower_trigger(self.trigger_mapping["experiment/end"])
                if log_file:
                    self.log_event(
                        event_time=time.perf_counter() - self.start_time,
                        block="experiment",
                        ISI="NA",
                        intensity="NA",
                        event_type="experiment/end",
                        trigger=self.trigger_mapping["experiment/end"],
                        n_in_block="NA",
                        correct="NA",
                        reset_QUEST=False,
                        rt="NA",
                        log_file=log_file
                    )

        self.listener.stop_listener()  # Stop the keyboard listener