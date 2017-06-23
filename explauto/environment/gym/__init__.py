from .gym_env import GymEnvironment
import numpy as np


def maximal_span_mountain_car(rollout):
    """Returns the max span of a rallout, typically the min and max positions in the Mountain Car environment."""
    # A rollout = Series of (position, speed)
    min_span = min(rollout[0, :])
    max_span = max(rollout[0, :])
    return((min_span, max_span))


def maximal_height_mountain_car(rollout):
    max_height = max(np.abs(rollout[0, :]))
    return([max_height])


environment = GymEnvironment
configurations = {'MCC_span': {"name": "MountainCarContinuous-v0", "observation_function": maximal_span_mountain_car, "s_mins": [-1.2, -1.2], "s_maxs": [0.6, 0.6]}, 'MCC_height': {"name": "MountainCarContinuous-v0", "observation_function": maximal_height_mountain_car, "s_mins": [-1.2], "s_maxs": [0.6]}}


def testcases(**kwargs):
    pass
