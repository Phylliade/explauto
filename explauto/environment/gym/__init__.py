from .gym_env import GymEnvironment
import numpy as np

MC_center = 0
# FIXME: Use an env_wrapper attribute
MC_max_pos = 0.6 - (-0.523)
MC_min_pos = -1.2 - (-0.523)


def maximal_span_mountain_car(rollout):
    """Returns the max span of a rallout, typically the min and max positions in the Mountain Car environment."""
    # A rollout = Series of (position, speed)
    min_span = min(rollout[:, 0] - MC_center)
    max_span = max(rollout[:, 0] - MC_center)
    return([min_span, max_span])


def maximal_height_mountain_car(rollout):
    # Center the position to the initial start
    max_height = max(np.abs(rollout[:, 0] - MC_center))
    return([max_height])


def minimal_position_mountain_car(rollout):
    minimum_position = min(rollout[:, 0] - MC_center)
    return([minimum_position])


def maximal_position_mountain_car(rollout):
    maximum_position = max(rollout[:, 0] - MC_center)
    return([maximum_position])


def energy_mountain_car(rollout):
    """Computes the energy consumed during a rollout"""
    # Use the energy formula as given by:
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py#L72
    action = rollout[:, 1]
    energy = np.sum(np.power(action, 2) * 0.1)
    return([energy])


def maximal_position_energy_mountain_car(rollout):
    """Use the energy and the maximal position"""
    energy = energy_mountain_car(rollout)[0]
    max_pos = maximal_position_mountain_car(rollout)[0]

    return([max_pos, energy])


environment = GymEnvironment
configurations = {
    'MCC_default': {"name": "MountainCarContinuous-v1", "observation_function": maximal_span_mountain_car, "s_mins": [MC_min_pos, MC_min_pos], "s_maxs": [MC_max_pos, MC_max_pos]},
    'MCC_span': {"name": "MountainCarContinuous-v1", "observation_function": maximal_span_mountain_car, "s_mins": [MC_min_pos, MC_min_pos], "s_maxs": [MC_max_pos, MC_max_pos]},
    'MCC_height': {"name": "MountainCarContinuous-v1", "observation_function": maximal_height_mountain_car, "s_mins": [0], "s_maxs": [max(abs(MC_max_pos), abs(MC_min_pos))]},
    'MCC_min_pos': {"name": "MountainCarContinuous-v1", "observation_function": minimal_position_mountain_car, "s_mins": [MC_min_pos], "s_maxs": [0.]},
    'MCC_max_pos': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos]},
    'MCC_max_pos_energy': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_energy_mountain_car, "s_mins": [0., 0.], "s_maxs": [MC_max_pos, 0.1]},
    'MCC_max_pos_tanh_unbiased': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos], "controler": "NN_tanh_unbiased"},
    'MCC_max_pos_tanh': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos], "controler": "NN_tanh"}
    # Maximal energy = 1 * 1000 * 0.1 = 100
}


def testcases(**kwargs):
    pass
