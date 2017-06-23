from .gym_env import GymEnvironment


def maximal_span_mountain_car(rollout):
    """Returns the max span of a rallout, typically the min and max positions in the Mountain Car environment."""
    min_span = min(rollout[0, :])
    max_span = max(rollout[0, :])
    return((min_span, max_span))


environment = GymEnvironment
configurations = {'MCC': {"name": "MountainCarContinuous-v0", "observation_function": maximal_span_mountain_car, "s_mins": [-1.2, -1.2], "s_maxs": [0.6, 0.6]}}


def testcases(**kwargs):
    pass
