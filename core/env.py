# core/env.py
# Defines the RL environment and Action enum for log triage.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum


class Action(IntEnum):
    IGNORE    = 0
    MONITOR   = 1
    ESCALATE  = 2


class LogTriageEnvironment(gym.Env):
    """
    A simple RL environment for DevOps log triage.
    
    Observation: a 10-dimensional float32 vector representing log features.
    Action:      one of {IGNORE, MONITOR, ESCALATE}
    Reward:      +1.0 for correct triage, 0.0 otherwise.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(len(Action))

        self._state      = None
        self._true_label = None
        self._step_count = 0
        self._max_steps  = 50

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._state      = self.observation_space.sample()
        self._true_label = int(self.np_random.integers(0, len(Action)))
        self._step_count = 0

        return self._state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        reward     = 1.0 if int(action) == self._true_label else 0.0
        self._step_count += 1

        terminated = self._step_count >= self._max_steps
        truncated  = False

        info = {
            "true_label" : Action(self._true_label).name,
            "agent_action": Action(int(action)).name,
            "step"        : self._step_count,
        }

        # Refresh state for next step
        self._state      = self.observation_space.sample()
        self._true_label = int(self.np_random.integers(0, len(Action)))

        return self._state.copy(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        pass

    def close(self):
        pass
