from .gym_env import GymEnvironment
from .mcc import MC_max_pos, MC_min_pos, maximal_span_mountain_car, maximal_height_mountain_car, maximal_position_mountain_car, maximal_position_energy_mountain_car, minimal_position_mountain_car
from .observation_functions import maximal_position


def sampled_rollout(rollout):
    return(rollout)

environment = GymEnvironment
configurations = {
    'MCC_default': {"name": "MountainCarContinuous-v1", "observation_function": maximal_span_mountain_car, "s_mins": [MC_min_pos, MC_min_pos], "s_maxs": [MC_max_pos, MC_max_pos]},
    'MCC_span': {"name": "MountainCarContinuous-v1", "observation_function": maximal_span_mountain_car, "s_mins": [MC_min_pos, MC_min_pos], "s_maxs": [MC_max_pos, MC_max_pos]},
    'MCC_height': {"name": "MountainCarContinuous-v1", "observation_function": maximal_height_mountain_car, "s_mins": [0], "s_maxs": [max(abs(MC_max_pos), abs(MC_min_pos))]},
    'MCC_min_pos': {"name": "MountainCarContinuous-v1", "observation_function": minimal_position_mountain_car, "s_mins": [MC_min_pos], "s_maxs": [0.]},
    'MCC_max_pos': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos]},
    'MCC_max_pos_energy': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_energy_mountain_car, "s_mins": [0., 0.], "s_maxs": [MC_max_pos, 0.1]},
    'MCC_max_pos_tanh_unbiased': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos], "controler": "NN_tanh_unbiased"},
    'MCC_max_pos_tanh': {"name": "MountainCarContinuous-v1", "observation_function": maximal_position_mountain_car, "s_mins": [0.], "s_maxs": [MC_max_pos], "controler": "NN_tanh"},
    "HalfCheetah": {"name": "HalfCheetah-v1", "observation_function": maximal_position, "s_mins": [0.], "s_maxs": [MC_max_pos], "controler": "NN_tanh"}
}


def testcases(**kwargs):
    pass
