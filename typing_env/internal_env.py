
import random
import logging
import torch
import numpy as np
from gym import spaces
from PIL import Image
import torchvision.transforms as transforms
from typing_env.kbd_env import KeyboardEnv
from setting import KEYS, PLACES, CHARS
from models.vision_encoder import VisionEncoder
from models.vision_agent import VisionAgent
from models.finger_agent import FingerAgent
from models.memory import Memory
from models.chord_memory import ChordMemory
from data.chord_vocab import CHORD_VOCAB
from data.sentences import Sentences
import pygame
from copy import copy
import jamspell
import string
from parameters import *
from config import DEFAULT_MODEL_DIR, DEFAULT_ROOT_DIR
import os.path as osp
from torchmetrics import CharErrorRate

jamspell_corrector = jamspell.TSpellCorrector()
jamspell_corrector.LoadLangModel('outputs/en.bin')
translator = str.maketrans('', '', string.punctuation)
ites = ["None", "JamSpell"] # "Norvig",


class InternalEnv(KeyboardEnv):
    def __init__(self, render_mode=None, img_folder='kbd1k/keyboard_dataset', position_file='kbd1k/keyboard_label.csv',
                 text_path='./data/sentences_fin.txt', vision_path='outputs/vision_agent.pt',
                 finger_path='outputs/finger_agent.pt', chars=CHARS, places=PLACES, keys=KEYS,
                 width=256, height=455, ite=False, finger_size=32, gaze_size=64,
                 parameters=None, two_thumb=False, chords=False, chords_active=False):
        """
        Args:
            render_mode: None or 'human'
            two_thumb: if True, use two-thumb typing with dynamic closest-finger assignment
                       and a no-crossing constraint (paper section 5.1.2). Both thumbs share
                       the same pre-trained finger agent weights.
            chords: if True, enable the hybrid chording scheme (requires two_thumb=True).
                    Extends action[1] to {0: type, 1: backspace, 2: chord}.
                    Adds 'chord_available', 'chord_recall_strength', and 'backspace_ready'
                    to the observation. The supervisor is saved separately as
                    supervisor_agent_two_thumb_chord.pt.
            chords_active: if True, chord obs signals are live and chord actions execute.
                    Requires chords=True. When False (default), all chord obs dimensions
                    return zero and chord actions demote to type — the model trains on the
                    full chord obs space but sees only the two-thumb typing problem.
                    Used for curriculum learning: train phase 1 with chords_active=False,
                    then continue training with chords_active=True.
        """
        if chords and not two_thumb:
            raise ValueError("chords=True requires two_thumb=True")
        if chords_active and not chords:
            raise ValueError("chords_active=True requires chords=True")

        super().__init__(render_mode, img_folder, position_file, width, height)
        self.logger = logging.getLogger(__name__)
        self.two_thumb = two_thumb
        self.chords = chords
        self.chords_active = chords_active

        if two_thumb:
            base_obs = {
                "certainty": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "correctness": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "P": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
                "vision_in_action": spaces.Discrete(2),
                # Which thumb is currently moving. Mutually exclusive — only one thumb
                # moves at a time. The supervisor learns to associate each side with
                # distinct position biases and expected movement durations.
                "left_finger_in_action": spaces.Discrete(2),
                "right_finger_in_action": spaces.Discrete(2),
            }
            if chords:
                base_obs["chord_available"] = spaces.Discrete(2)
                # Continuous recall strength so the supervisor can gauge confidence.
                base_obs["chord_recall_strength"] = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
                # backspace_ready: 1 when errors exist AND both fingers are idle.
                # Gives the supervisor a clean unambiguous signal for when backspace
                # is both meaningful and executable (per CLAUDE.md lessons learned).
                base_obs["backspace_ready"] = spaces.Discrete(2)
            self.observation_space = spaces.Dict(base_obs)
        else:
            self.observation_space = spaces.Dict(
                {
                    "certainty": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                    "correctness": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                    "P": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
                    "vision_in_action": spaces.Discrete(2),
                    "finger_in_action": spaces.Discrete(2),
                }
            )

        """
        Action space:
        goals for vision agent: 0: look at finger target for visual guidance; 1: look at input box for proofreading
        goals for finger agent: 0: type next character; 1: delete last character; 2: attempt chord (chords mode only)
        speeds for finger agent
        """
        if chords:
            self.action_space = spaces.MultiDiscrete([2, 3, len(SPEEDS)])
        else:
            self.action_space = spaces.MultiDiscrete([2, 2, len(SPEEDS)])

        self.vision_goal = None
        self.finger_goal = None
        self.speed = 0

        # text information
        self.target_text = "hello"
        self.typed_text = ''
        self.chars = chars
        self.places = places
        self.keys = keys

        # sub-agents
        self.foveal_encoder = VisionEncoder()
        self.foveal_encoder.load_state_dict(
            torch.load(osp.join(DEFAULT_MODEL_DIR, 'f_encoder.pt'), map_location=torch.device('cpu')))
        self.foveal_encoder.eval()
        self.peripheral_encoder = VisionEncoder()
        self.peripheral_encoder.load_state_dict(
            torch.load(osp.join(DEFAULT_MODEL_DIR, 'p_encoder.pt'), map_location=torch.device('cpu')))
        self.peripheral_encoder.eval()
        self.vision_agent = VisionAgent(load=vision_path)

        if two_thumb:
            # Both thumbs use the same pre-trained weights; two independent instances are
            # needed so each carries its own internal state (e.g. numpy RNG).
            self.left_finger_agent  = FingerAgent(load=finger_path)
            self.right_finger_agent = FingerAgent(load=finger_path)
        else:
            self.finger_agent = FingerAgent(load=finger_path)

        self.wm = Memory(model_path=osp.join(DEFAULT_MODEL_DIR, 'wm_encoder.pt'))
        self.wm.target(self.target_text)

        if chords:
            self.chord_memory = ChordMemory()

        """ initialize the environment """
        self.gaze_size = gaze_size
        self._randomize_gaze()
        self.finger_size = finger_size
        self._randomize_finger()
        img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot)
        img_tensor = transforms.ToTensor()(img)
        self.peripheral_z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()
        self.sentence_db = Sentences(load_path=text_path)
        self.log = []
        self.ep_len = 0
        self.vision_in_action = False
        self.finger_movement_time = 0
        self.gaze_movement_time = 0
        self.last_proofreading_time = 0
        self.last_guiding_time = 0
        self.ite = ite
        self.cer = CharErrorRate()

        """ statistical data """
        self.immediate_backspaces = 0
        self.delayed_backspaces = 0
        self.incorrect_backspaces = 0
        self.correct_type_keys = 0
        self.incorrect_type_keys = 0
        self.chords_fired = 0
        self.chord_attempts = 0

        """ initialize the parameters """
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = [0.5, 0.5, 0.5]

    # ------------------------------------------------------------------
    # Two-thumb helper
    # ------------------------------------------------------------------

    def _assign_finger(self, key):
        """Return which thumb ('left' or 'right') should press key.

        The thumb closest to the key's center is assigned, subject to the
        no-crossing constraint from paper section 5.1.2: the left thumb must
        not end up to the right of the right thumb's current x-position, and
        vice versa.
        """
        key_pos = np.array(self._get_center(key))
        d_left  = self._distance(self.left_finger,  key_pos)
        d_right = self._distance(self.right_finger, key_pos)

        key_x = key_pos[0]

        if d_left <= d_right:
            # Left thumb is closer. Only assign if doing so would not place
            # the left thumb to the right of the right thumb.
            if key_x <= self.right_finger[0]:
                return 'left'
            return 'right'
        else:
            # Right thumb is closer. Only assign if doing so would not place
            # the right thumb to the left of the left thumb.
            if key_x >= self.left_finger[0]:
                return 'right'
            return 'left'

    # ------------------------------------------------------------------
    # Chord helpers
    # ------------------------------------------------------------------

    def _chord_obs_bits(self):
        """Compute chord_available, chord_recall_strength, and backspace_ready.

        chord_available is 1 only when:
          - a chord expansion matches the start of the remaining target text
            based on physical typed_text (not wm.recall_text),
          - the last physically-typed character is correct (no pending error), and
          - both fingers are idle (no movement in progress).

        Using typed_text for position prevents false positives when WM drifts:
        if WM forgets everything (recall_text=''), correctness() returns 1.0 and
        next_chord('', target) finds a chord at position 0 — even though
        typed_text already contains several characters.  Using typed_text makes
        the chord position reflect physical reality, not cognitive recall.

        backspace_ready is 1 when:
          - the most recently typed character is actually wrong (typed_text[-1]
            does not match the corresponding target character), AND
          - both fingers are idle.

        Crucially, this checks physical typed_text, NOT wm.correctness().
        WM correctness reflects recall uncertainty (which decays over time even
        when every character is correct) so using it produces spurious backspace
        signals on perfectly correct text.  The misfire case —
          intended='u' tapped='u' | correctness=0.692 | typed='the compu' (correct)
        — has correctness < 1 purely because WM forgot the first character, not
        because of any typing error.  Checking typed_text directly avoids this.
        """
        # In phase 1 of curriculum learning (chords_active=False), all chord obs
        # signals are zeroed so the model trains purely on two-thumb typing.
        if not self.chords_active:
            return 0, 0.0, 0

        both_idle = not self.left_finger_in_action and not self.right_finger_in_action

        # Last typed character is actually wrong (not just uncertain in WM).
        last_char_wrong = (
            len(self.typed_text) > 0
            and len(self.typed_text) <= len(self.target_text)
            and self.typed_text[-1] != self.target_text[len(self.typed_text) - 1]
        )

        # Chord position uses typed_text (physical truth) so that WM drift
        # — where recall_text lags or forgets typed_text — does not produce
        # false chord opportunities on already-typed or corrupted prefixes.
        current_chord = self.chord_memory.next_chord(self.typed_text, self.target_text)
        chord_available = int(
            current_chord is not None
            and not last_char_wrong
            and both_idle
        )
        recall_str = self.chord_memory.recall_strength(current_chord) if current_chord is not None else 0.0

        backspace_ready = int(last_char_wrong and both_idle)
        return chord_available, recall_str, backspace_ready

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        if self.two_thumb:
            obs = {
                "correctness":             [self.wm.correctness()],
                "certainty":               [self.wm.memory_certainty],
                "P":                       [self.parameters[0], self.parameters[1], self.parameters[2]],
                "vision_in_action":        1 if self.vision_in_action else 0,
                "left_finger_in_action":   1 if self.left_finger_in_action else 0,
                "right_finger_in_action":  1 if self.right_finger_in_action else 0,
            }
            if self.chords:
                chord_available, recall_str, backspace_ready = self._chord_obs_bits()
                obs["chord_available"] = chord_available
                obs["chord_recall_strength"] = [recall_str]
                obs["backspace_ready"] = backspace_ready
            return obs
        return {
            "correctness":      [self.wm.correctness()],
            "certainty":        [self.wm.memory_certainty],
            "P":                [self.parameters[0], self.parameters[1], self.parameters[2]],
            "vision_in_action": 1 if self.vision_in_action else 0,
            "finger_in_action": 1 if self.finger_in_action else 0,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        self.ep_len += 1
        self._chord_executed = False

        """ set finger """
        if self.chords and self.chords_active and action[1] == 2:
            # Chord path: attempt only when both fingers are idle.
            # If a finger is in flight, fall back to type to avoid blocking.
            if not self.left_finger_in_action and not self.right_finger_in_action:
                reward_shaping = self._step_chord(action)
                self._chord_executed = True
            else:
                # Fingers busy — silently demote to type action.
                type_action = [action[0], 0, action[2]]
                self._step_finger_two_thumb(type_action)
        elif self.two_thumb:
            if self.chords and not self.chords_active and action[1] == 2:
                # Phase 1 curriculum: chord signals are zeroed so the supervisor
                # should rarely pick action[1]=2, but if it does, demote to type.
                type_action = [action[0], 0, action[2]]
                self._step_finger_two_thumb(type_action)
            else:
                self._step_finger_two_thumb(action)
        else:
            self._step_finger_single(action)

        """ set vision """
        if not self.vision_in_action:
            if action[0] == 0 and not self._gaze_on(self.finger_goal):
                self.vision_goal = self.finger_goal
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = False
                if self.vision_goal:
                    self.rollout_vision()
                    self.vision_in_action = True
            elif action[0] == 0 and self._gaze_on(self.finger_goal):
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = False
                self.vision_in_action = True
            elif action[0] == 1 and self.vision_goal != 'input_box':
                self.vision_goal = 'input_box'
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = True
                self.rollout_vision()
                self.vision_in_action = True
            elif action[0] == 1 and self._gaze_on('input_box'):
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = True
                self.vision_in_action = True
        else:
            if self.gaze_movement_time <= 0 and self.vision_goal:
                self.vision_in_action = False

        """ run vision and gaze """
        self.gaze_movement_time -= TIMESTEP_TIME
        if not self._chord_executed:
            self.finger_movement_time -= TIMESTEP_TIME

        if not self._chord_executed:
            reward_shaping = 0
        reward_shaping_value = 0.08

        # Chord execution already set is_tapping inside _step_chord(); only
        # reset it here for the sequential (non-chord) path.
        if not self._chord_executed:
            self.is_tapping = False
        if not self._chord_executed:
            if self.finger_in_action and self.finger_movement_time <= 0:
                self.is_tapping, _ = self.rollout_finger()
                if self.is_tapping and self._where():
                    reward_shaping = self._process_keypress(reward_shaping, reward_shaping_value)
                self.finger_in_action = False
                if self.two_thumb:
                    if self.active_finger == 'left':
                        self.left_finger_in_action = False
                    else:
                        self.right_finger_in_action = False

        """ update memory """
        # Chord execution updates WM internally; skip here to avoid double-encoding.
        if not self._chord_executed:
            self.update_memory()

        """ get observation """
        observation = self._get_obs()

        """ terminate or not """
        done = self.ep_len >= MAX_STEPS or (self._finger_on(">") and self.is_tapping)

        """ get reward """
        reward = self.reward(done) + reward_shaping

        """ log """
        self.log.append({
            "vision_goal":    self.vision_goal,
            "finger_goal":    self.finger_goal,
            "speed":          self.speed,
            "gaze":           copy(self.gaze),
            "finger":         copy(self.finger),  # active thumb position; used by Metrics
            "is_proofreading": self._gaze_on("input_box"),
            "is_tapping":     self.is_tapping,
            "target_text":    self.target_text,
            "tapped_key":     self._where(),
            "typed_text":     self.typed_text,
            "chord_executed": self._chord_executed,
        })
        info = {}
        if done and self.chords:
            info["chord_attempts"] = self.chord_attempts
            info["chords_fired"] = self.chords_fired
        return observation, reward, done, info

    def _step_finger_single(self, action):
        """Plan and time the single-thumb movement."""
        if not self.finger_in_action:
            if action[1] == 0:
                self.finger_goal = self.wm.next_char()
            elif action[1] == 1:
                self.finger_goal = "<"
            self.speed = SPEEDS[action[2]]

            f_obs = {
                "pixels":  self.peripheral_z,
                "finger":  self.finger / np.array([self.width, self.height]),
                "tapping": 1 if self.is_tapping else 0,
                "target":  self.keys.index(self.finger_goal),
            }
            pos, _ = self.finger_agent.predict(f_obs, deterministic=True)
            next_pos = [pos[0], pos[1] + int(self.height / 2)]
            distance = self._distance(self.finger, next_pos)

            movement_time = round(distance / self.speed)
            if movement_time < TAPPING_TIME:
                movement_time = TAPPING_TIME
            self.error = error_distance(movement_time, distance, self.parameters[1], with_gaze=True)
            self.finger_movement_time = movement_time
            self.finger_in_action = True

    def _step_finger_two_thumb(self, action):
        """Plan and time a two-thumb movement using dynamic closest-finger assignment.

        Only one thumb moves at a time (matching the single-supervisor-goal architecture).
        The thumb closest to the target key is selected, subject to the no-crossing constraint.
        """
        if not self.finger_in_action:
            if action[1] == 0:
                self.finger_goal = self.wm.next_char()
            elif action[1] == 1:
                self.finger_goal = "<"
            self.speed = SPEEDS[action[2]]

            self.active_finger = self._assign_finger(self.finger_goal)
            active_pos = self.left_finger if self.active_finger == 'left' else self.right_finger

            f_obs = {
                "pixels":  self.peripheral_z,
                "finger":  active_pos / np.array([self.width, self.height]),
                "tapping": 1 if self.is_tapping else 0,
                "target":  self.keys.index(self.finger_goal),
            }
            agent = self.left_finger_agent if self.active_finger == 'left' else self.right_finger_agent
            pos, _ = agent.predict(f_obs, deterministic=True)
            next_pos = [pos[0], pos[1] + int(self.height / 2)]
            distance = self._distance(active_pos, next_pos)

            movement_time = round(distance / self.speed)
            if movement_time < TAPPING_TIME:
                movement_time = TAPPING_TIME
            self.error = error_distance(movement_time, distance, self.parameters[1], with_gaze=True)
            self.finger_movement_time = movement_time
            self.finger_in_action = True

            if self.active_finger == 'left':
                self.left_finger_in_action  = True
                self.right_finger_in_action = False
            else:
                self.right_finger_in_action = True
                self.left_finger_in_action  = False

    def _step_chord(self, action):
        """Atomic chord execution: plan, apply noise, resolve, update state.

        The chord is treated as a single supervisor decision with immediate
        resolution — no in-flight period, no blocked timesteps.

        Execution flow:
          1. Identify which chord to attempt from ChordMemory.
          2. If no chord is available (bad position or low correctness), fall
             back to a type action and return a small penalty.
          3. Assign left/right key by x-position.
          4. Compute movement times from current finger positions.
          5. Apply independent Gaussian motor noise to each landing position.
          6. Timing check: |mt_left - mt_right| <= CHORD_WINDOW_MS.
          7. Motor check: did each finger land on its assigned key?
          8. Success → append expansion; failure → append individual landed chars.
          9. Update WM and ChordMemory; return reward shaping value.

        Returns:
            reward_shaping (float): positive on chord success, negative on failure
                                    or invalid attempt.
        """
        reward_shaping_value = 0.08

        # Use typed_text for position to stay in sync with _chord_obs_bits.
        chord_key = self.chord_memory.next_chord(self.typed_text, self.target_text)

        # Guard: invalid chord attempt (no match or errors exist).
        # last_char_wrong check mirrors the gate in _chord_obs_bits so that
        # a chord attempt demoted by _chord_obs_bits is also demoted here.
        last_char_wrong = (
            len(self.typed_text) > 0
            and len(self.typed_text) <= len(self.target_text)
            and self.typed_text[-1] != self.target_text[len(self.typed_text) - 1]
        )
        if chord_key is None or last_char_wrong:
            # Demote to type; penalise for attempting chord when unavailable.
            type_action = [action[0], 0, action[2]]
            self._step_finger_two_thumb(type_action)
            return -reward_shaping_value

        self.chord_attempts += 1
        # Attempt bonus: fires for every valid chord attempt regardless of success.
        # Creates a direct gradient signal connecting chord_available=1 to choosing
        # action[1]=2. Without this, the model only sees chord reward on the rare
        # occasions it accidentally explores chord, which is too sparse to learn from.
        chord_attempt_bonus = 0.15
        expansion = CHORD_VOCAB[chord_key]
        k1, k2 = chord_key  # sorted tuple

        # Assign keys to thumbs by x-position so no-crossing is respected.
        center1 = self._get_center(k1)
        center2 = self._get_center(k2)
        if center1[0] <= center2[0]:
            left_key,  left_center  = k1, center1
            right_key, right_center = k2, center2
        else:
            left_key,  left_center  = k2, center2
            right_key, right_center = k1, center1

        self.speed = SPEEDS[action[2]]

        # Movement times from current thumb positions to chord key centers.
        dist_left  = self._distance(self.left_finger,  left_center)
        dist_right = self._distance(self.right_finger, right_center)
        mt_left  = max(TAPPING_TIME, round(dist_left  / self.speed))
        mt_right = max(TAPPING_TIME, round(dist_right / self.speed))

        # Motor noise (same model as sequential taps).
        err_left  = error_distance(mt_left,  dist_left,  self.parameters[1], with_gaze=True)
        err_right = error_distance(mt_right, dist_right, self.parameters[1], with_gaze=True)

        new_left = np.array([
            left_center[0]  + np.random.normal(0, err_left),
            left_center[1]  + np.random.normal(0, err_left),
        ], dtype=float)
        new_right = np.array([
            right_center[0] + np.random.normal(0, err_right),
            right_center[1] + np.random.normal(0, err_right),
        ], dtype=float)

        # Update physical finger positions.
        self.left_finger  = new_left
        self.right_finger = new_right

        # Timing check: both taps must land within CHORD_WINDOW_MS of each other.
        timing_ok = abs(mt_left - mt_right) <= CHORD_WINDOW_MS

        # Motor check: temporarily point self.finger at each thumb to use _where().
        self.finger = self.left_finger
        left_landed = self._where()

        self.finger = self.right_finger
        right_landed = self._where()

        # Restore self.finger to the right (dominant) thumb.
        self.finger = self.right_finger
        self.finger_goal = right_key  # used by logging and vision system
        self.is_tapping = True

        # WM time decay (mirrors update_memory path for sequential taps).
        time_since_proofread = self.ep_len * TIMESTEP_TIME - self.last_proofreading_time
        self.wm.forget(time=time_since_proofread, parameter=self.parameters[0])

        chord_fired = (
            timing_ok
            and left_landed  == left_key
            and right_landed == right_key
        )

        if chord_fired:
            # Append full expansion and advance WM for each character.
            for ch in expansion:
                self.typed_text += ch
                self.wm.encode_key(ch)
            self.chord_memory.reinforce(chord_key)
            self.chords_fired += 1
            # Reward = chars produced + bonus for each supervisor step saved.
            # A chord typing N chars uses 1 supervisor step instead of N, saving
            # (N - 1) steps.  Each saved step earns the same reward_shaping_value
            # as a correct sequential tap, making chording strictly better than
            # sequential whenever it succeeds.
            #   "the"  (3 chars, saves 2): 0.08*3 + 0.08*2 = 0.40  vs sequential 0.24
            #   "with" (4 chars, saves 3): 0.08*4 + 0.08*3 = 0.56  vs sequential 0.32
            saved_steps = len(expansion) - 1
            return chord_attempt_bonus + reward_shaping_value * len(expansion) + reward_shaping_value * saved_steps
        else:
            # Fallback: register wherever each finger actually landed.
            reward_shaping = chord_attempt_bonus
            for landed, intended in [(left_landed, left_key), (right_landed, right_key)]:
                if landed and landed not in ('<', '>'):
                    self.typed_text += landed
                    self.wm.encode_key(landed)
                    if landed == intended:
                        reward_shaping += 0.0   # neutral: got the char but not the chord
                    else:
                        reward_shaping -= reward_shaping_value  # misfire
                else:
                    reward_shaping -= reward_shaping_value
            return reward_shaping

    # ------------------------------------------------------------------
    # Key-press processing (shared between single and two-thumb paths)
    # ------------------------------------------------------------------

    def _process_keypress(self, reward_shaping, reward_shaping_value):
        """Update typed text and reward shaping based on the tapped key."""
        key = self._where()

        if self.finger_goal == "<" and key == "<":
            if self.typed_text == '':
                self.incorrect_backspaces += 1
                reward_shaping -= reward_shaping_value
            elif len(self.typed_text) <= len(self.target_text):
                if self.typed_text[-1] != self.target_text[len(self.typed_text) - 1]:
                    self.immediate_backspaces += 1
                    reward_shaping += reward_shaping_value
                elif self.typed_text != self.target_text[0:len(self.typed_text)]:
                    self.delayed_backspaces += 1
                    reward_shaping += reward_shaping_value / 2
                else:
                    self.incorrect_backspaces += 1
                    reward_shaping -= reward_shaping_value
            else:
                self.immediate_backspaces += 1
                reward_shaping += reward_shaping_value
        else:
            if key == self.finger_goal:
                self.correct_type_keys += 1
                reward_shaping += reward_shaping_value
            else:
                self.incorrect_type_keys += 1
                reward_shaping -= reward_shaping_value

        if key == '<':
            if self.typed_text != '':
                self.typed_text = self.typed_text[:-1]
        elif key == ' ':
            self.typed_text += ' '
        elif key == '>':
            self.typed_text += ''
        else:
            self.typed_text += key

        return reward_shaping

    # ------------------------------------------------------------------
    # Finger rollout
    # ------------------------------------------------------------------

    def rollout_finger(self):
        """Execute the finger agent for the current goal and apply motor noise.

        In two-thumb mode, dispatches to the correct thumb's agent and updates
        that thumb's position. self.finger is always synced to the active
        thumb so that _finger_on() / _where() in the parent class work correctly.
        """
        if self.two_thumb:
            active_pos = self.left_finger if self.active_finger == 'left' else self.right_finger
            agent      = self.left_finger_agent if self.active_finger == 'left' else self.right_finger_agent
        else:
            active_pos = self.finger
            agent      = self.finger_agent

        f_obs = {
            "pixels":  self.peripheral_z,
            "finger":  active_pos / np.array([self.width, self.height]),
            "tapping": 1 if self.is_tapping else 0,
            "target":  self.keys.index(self.finger_goal),
        }
        action, _ = agent.predict(f_obs, deterministic=True)
        is_tapping, _ = self._action_to_pot(action)

        new_x = action[0]
        new_y = action[1] + int(self.height / 2)

        self.is_guiding_fingers = self._gaze_on(place=self.finger_goal)
        if self.is_guiding_fingers:
            self.last_guiding_time = self.ep_len * TIMESTEP_TIME
        time_since_guide = self.ep_len * TIMESTEP_TIME - self.last_guiding_time

        new_x += self.finger_noise_distance(wo_gaze_time=time_since_guide)
        new_y += self.finger_noise_distance(wo_gaze_time=time_since_guide)

        if self.two_thumb:
            if self.active_finger == 'left':
                self.left_finger[0] = new_x
                self.left_finger[1] = new_y
            else:
                self.right_finger[0] = new_x
                self.right_finger[1] = new_y
            # Sync self.finger to the active thumb so the parent-class
            # _finger_on() and _where() methods see the correct position.
            self.finger = self.left_finger if self.active_finger == 'left' else self.right_finger
        else:
            self.finger[0] = new_x
            self.finger[1] = new_y

        return is_tapping, None

    def finger_noise_distance(self, wo_gaze_time):
        if wo_gaze_time > 0:
            return np.random.normal(0, self.error + error_wo_gaze(wo_gaze_time))
        return np.random.normal(0, self.error)

    # ------------------------------------------------------------------
    # Vision rollout
    # ------------------------------------------------------------------

    def rollout_vision(self):
        x = self.gaze[0]
        y = self.gaze[1]
        left, top, right, bottom = (x - self.gaze_size / 2, y - self.gaze_size / 2,
                                    x + self.gaze_size / 2, y + self.gaze_size / 2)
        img = self.screenshot.crop((left, top, right, bottom))
        img_tensor = transforms.ToTensor()(img)
        foveal_z = torch.squeeze(self.foveal_encoder(img_tensor)).detach().numpy()
        v_obs = {
            "gaze":       self.gaze / np.array([self.width, self.height]),
            "foveal":     foveal_z,
            "peripheral": self.peripheral_z,
            "target":     self.places.index(self.vision_goal),
        }
        action, _ = self.vision_agent.predict(v_obs, deterministic=True)
        movement = self._action_to_movement(action)
        self.gaze[0] = action[0]
        self.gaze[1] = action[1]
        self.gaze[0] += np.random.normal(0, 5)
        self.gaze[1] += np.random.normal(0, 5)
        return movement

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def update_memory(self):
        if self.is_proofreading:
            self.last_proofreading_time = self.ep_len * TIMESTEP_TIME
            self.wm.proofread(typed_text=self.typed_text)
        elif self.is_tapping and self._where():
            time = self.ep_len * TIMESTEP_TIME - self.last_proofreading_time
            self.wm.forget(time=time, parameter=self.parameters[0])
            if self.chords:
                self.chord_memory.decay_all(elapsed_ms=time, lambda_param=self.parameters[0])
            if self._gaze_on_finger():
                vision_obs = self.peripheral_z
                finger_obs = self.finger / np.array([self.width, self.height])
                x = torch.from_numpy(np.concatenate((vision_obs, finger_obs))).float()
                self.wm.encode(x)
            else:
                self.wm.encode_key(key=self.finger_goal)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, parameters=None, gaze=None, finger=None, target_text=None, reset_kbd=False):
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        self.ep_len = 0
        if reset_kbd:
            self._reset_kbd()
            img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot)
            img_tensor = transforms.ToTensor()(img)
            self.peripheral_z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()
        if gaze is not None:
            self.gaze = gaze
        else:
            self._randomize_gaze()
        if finger is not None:
            self.finger = finger
            self.is_tapping = False
        else:
            self._randomize_finger()
        if target_text is not None:
            self.target_text = target_text
        else:
            self.target_text = self.sentence_db.random_sentence()
        while len(self.target_text) == 0:
            self.target_text = self.sentence_db.random_sentence()
        self.wm.target(self.target_text)
        if self.chords:
            self.chord_memory.reset()
        self.typed_text = ''
        self.finger_movement_time = 0
        self.gaze_movement_time = 0
        self.last_proofreading_time = 0
        self.last_guiding_time = 0

        self.immediate_backspaces = 0
        self.delayed_backspaces = 0
        self.incorrect_backspaces = 0
        self.correct_type_keys = 0
        self.incorrect_type_keys = 0
        self.chords_fired = 0
        self.chord_attempts = 0

        self.log = []
        self.log.append({
            "vision_goal":    None,
            "finger_goal":    None,
            "speed":          0,
            "gaze":           copy(self.gaze),
            "is_proofreading": False,
            "finger":         copy(self.finger),
            "is_tapping":     self.is_tapping,
            "tapped_key":     "",
            "typed_text":     self.typed_text,
            "chord_executed": False,
        })
        return self._get_obs()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def reward(self, done):
        r = 0
        if done:
            r += (1 - self.cer(self.typed_text, self.target_text).item() ** 0.4)
            r -= 1 * time_penalty(self.target_text, self.ep_len * TIMESTEP_TIME)
        return r

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        else:
            print("=====================================")
            print("finger goal: ", self.finger_goal)
            print("target text: ", self.target_text)
            print("typed text:", self.typed_text)
            print("recall text:", self.wm.recall())
            if self.chords:
                chord_key = self.chord_memory.next_chord(self.wm.recall_text, self.target_text)
                if chord_key:
                    print(f"chord available: {chord_key} → '{CHORD_VOCAB[chord_key]}' "
                          f"(recall={self.chord_memory.recall_strength(chord_key):.2f})")

    def _render_frame(self):
        render_fps = 20
        resize_factor = 2
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('crtypist')
            self.window = pygame.display.set_mode((self.width * resize_factor, self.height * resize_factor))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        foveated_image = Image.open(self.img_names[self.img_index])
        foveated_image = transforms.Resize((self.height * resize_factor, self.width * resize_factor))(foveated_image)
        mode = foveated_image.mode
        size = foveated_image.size
        data = foveated_image.tobytes()
        image = pygame.image.fromstring(data, size, mode)

        font = pygame.font.SysFont(None, 40)
        field_text  = font.render(self.typed_text, True, 'black')
        target_text = font.render("target: %s" % (self.target_text), True, 'gray')
        recall_text = font.render("memory: %s" % (self.wm.recall()), True, 'black')

        self.window.blit(image, (0, 0))
        self.window.blit(field_text, (20, 50))

        if self.two_thumb:
            self._render_thumb(self.left_finger,  'left',  resize_factor)
            self._render_thumb(self.right_finger, 'right', resize_factor)
        else:
            if self.is_tapping:
                pygame.draw.circle(self.window, 'blue',
                                   (self.finger[0] * resize_factor, self.finger[1] * resize_factor),
                                   16 * resize_factor, 0)
            elif self.finger_in_action:
                pygame.draw.circle(self.window, 'blue',
                                   (self.finger[0] * resize_factor, self.finger[1] * resize_factor),
                                   16 * resize_factor, 3)
            else:
                pygame.draw.circle(self.window, 'gray',
                                   (self.finger[0] * resize_factor, self.finger[1] * resize_factor),
                                   16 * resize_factor, 3)

        if self.gaze[1] < 200:
            self.gaze[0] = 40
        pygame.draw.rect(self.window, 'red',
                         ((self.gaze[0] - self.gaze_size / 2) * resize_factor,
                          (self.gaze[1] - self.gaze_size / 2) * resize_factor,
                          self.gaze_size * resize_factor,
                          self.gaze_size * resize_factor), 3)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(render_fps)

    def _render_thumb(self, finger_pos, side, resize_factor):
        """Draw a single thumb on the pygame surface.

        Left thumb: blue.  Right thumb: orange.
        Filled circle while tapping, outline otherwise.
        Active (moving) thumb shown with full opacity; idle thumb shown lighter.
        """
        is_active = (self.active_finger == side) and self.finger_in_action
        is_this_tapping = self.is_tapping and (self.active_finger == side)

        if side == 'left':
            color_active = 'blue'
            color_idle   = 'cornflowerblue'
        else:
            color_active = 'orange'
            color_idle   = 'moccasin'

        color = color_active if is_active else color_idle
        width = 0 if is_this_tapping else 3

        pygame.draw.circle(
            self.window, color,
            (int(finger_pos[0] * resize_factor), int(finger_pos[1] * resize_factor)),
            16 * resize_factor,
            width,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _randomize_gaze(self):
        self.vision_goal      = None
        self.gaze             = np.array([127, 300])
        self.is_proofreading  = False
        self.is_guiding_fingers = False
        self.vision_in_action = False

    def _randomize_finger(self):
        self.finger_goal    = None
        self.is_tapping     = False
        self.finger_in_action = False

        if self.two_thumb:
            # Left thumb starts in the left quarter; right in the right quarter.
            # Centering each thumb over its natural half of the keyboard gives
            # balanced travel distances during training, which produces a stronger
            # proofreading reward signal and prevents right-finger dominance.
            self.left_finger  = np.array([64,  454], dtype=float)
            self.right_finger = np.array([192, 454], dtype=float)
            self.active_finger = 'right'
            self.left_finger_in_action  = False
            self.right_finger_in_action = False
            # Keep self.finger pointing at the right (dominant) thumb so that
            # any parent-class calls before the first step are well-defined.
            self.finger = self.right_finger
        else:
            self.finger = np.array([255, 454], dtype=float)

    # ------------------------------------------------------------------
    # Action conversion helpers
    # ------------------------------------------------------------------

    def _action_to_movement(self, action):
        x = action[0]
        y = action[1]
        return np.array([x - self.gaze[0], y - self.gaze[1]])

    def _action_to_pot(self, action, noise=False):
        is_tapping = action[2] == 0
        return is_tapping, None
