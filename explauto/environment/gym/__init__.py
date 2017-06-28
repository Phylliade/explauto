from .gym_env import GymEnvironment
import numpy as np


def maximal_span_mountain_car(rollout):
    """Returns the max span of a rallout, typically the min and max positions in the Mountain Car environment."""
    # A rollout = Series of (position, speed)
    min_span = min(rollout[:, 0] - (-0.523))
    max_span = max(rollout[:, 0] - (-0.523))
    return((min_span, max_span))


def maximal_height_mountain_car(rollout):
    # Center the position to the initial start
    max_height = max(np.abs(rollout[:, 0] - (-0.523)))
    return([max_height])


def minimal_position_mountain_car(rollout):
    minimum_position = min(rollout[:, 0] - (-0.523))
    return([minimum_position])


def maximal_position_mountain_car(rollout):
    maximum_position = max(rollout[:, 0] - (-0.523))
    return([maximum_position])


environment = GymEnvironment
configurations = {
    'MCC_default': {"name": "MountainCarContinuous-v0", "observation_function": maximal_span_mountain_car, "s_mins": [-1.2, -1.2], "s_maxs": [0.6, 0.6]},
    'MCC_span': {"name": "MountainCarContinuous-v0", "observation_function": maximal_span_mountain_car, "s_mins": [-1.2, -1.2], "s_maxs": [0.6, 0.6]},
    'MCC_height': {"name": "MountainCarContinuous-v0", "observation_function": maximal_height_mountain_car, "s_mins": [0], "s_maxs": [1.2]},
    'MCC_min_pos': {"name": "MountainCarContinuous-v0", "observation_function": minimal_position_mountain_car, "s_mins": [-1.2], "s_maxs": [0.]},
    'MCC_max_pos': {"name": "MountainCarContinuous-v0", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [1.2]}
}


def testcases(**kwargs):
    pass
