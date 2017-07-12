from ..environment import Environment
from .controler import NNControler
import gym as openai_gym
import numpy as np
import pickle


DEBUG = False


class GymEnvironment(Environment):
    def __init__(self, name, observation_function=None, s_mins=None, s_maxs=None, controler="NN_unbiased"):
        env = openai_gym.make(name)

        # env = openai_gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        self.env = env
        self.env.seed(123)
        self.last_observation = self.env.reset()
        param_min = -1000000
        param_max = 1000000

        if len(self.env.observation_space.shape) != 1 or len(self.env.action_space.shape) != 1:
            raise(ValueError("The action or observation space have more than one dimensions, which is not currently supported"))
        self.action_space_dim = self.env.action_space.shape[0]

        self.observation_space_dim = self.env.observation_space.shape[0]

        if controler in ["NN", "NN_tanh", "NN_unbiased", "NN_tanh_unbiased"]:
            if controler == "NN":
                self.controler = NNControler(tanh=False, bias=True, observation_space_dim=self.observation_space_dim, action_space_dim=self.action_space_dim)
            if controler == "NN_tanh":
                self.controler = NNControler(tanh=True, bias=True, observation_space_dim=self.observation_space_dim, action_space_dim=self.action_space_dim)
            elif controler == "NN_unbiased":
                self.controler = NNControler(tanh=False, bias=False, observation_space_dim=self.observation_space_dim, action_space_dim=self.action_space_dim)
            elif controler == "NN_tanh_unbiased":
                self.controler = NNControler(bias=False, tanh=True, observation_space_dim=self.observation_space_dim, action_space_dim=self.action_space_dim)

            m_mins = [param_min] * self.controler.parameter_space_dim
            m_maxs = [param_max] * self.controler.parameter_space_dim

        self.observation_function = (lambda rollout: rollout.flatten()) if observation_function is None else observation_function

        self.rollout_size = 1000

        self.replay_buffer = []

        # Automatically compute the size of the observation space only if the observation function is the identity
        if s_mins is None:
            s_mins = np.tile(self.env.observation_space.low, self.rollout_size)
        if s_maxs is None:
            s_maxs = np.tile(self.env.observation_space.high, self.rollout_size)

        Environment.__init__(
            self,
            m_mins,
            m_maxs,
            s_mins,
            s_maxs
        )

    def compute_motor_command(self, actions):
        if len(actions) != self.rollout_size:
            raise(ValueError("The size of actions ({}) does not match the rollout_size ({})").format(len(actions), self.rollout_size))

        return(actions)

    def compute_sensori_effect(self, controler_parameters, render=False, save_to_replay_buffer=False):
        # Start a new episode
        self.last_observation = self.env.reset()

        # We store a rallout as a series of states
        # Fill the rollout with the initial position, in case the episode stops before the end
        rollout = np.tile(self.last_observation, (self.rollout_size, 1))

        for step in range(1, self.rollout_size):
            self.controler.set_parameters(controler_parameters)
            action = self.controler(self.last_observation)

            observation, reward, done, info = self.env.step(action)
            rollout[step, :] = observation.squeeze()

            terminal = (step == (self.rollout_size - 1))
            if save_to_replay_buffer:
                self.replay_buffer.append([self.last_observation, action, reward, observation, terminal])

            self.last_observation = observation

            if render:
                self.env.render()

            if done:
                if DEBUG:
                    print("Done!")
                    print("Number of steps: {}".format(step))
                break

        observation = self.observation_function(rollout)
        return(observation)

    def save_replay_buffer(self, file="replay_buffer.p"):
        with open(file, "wb") as fd:
            pickle.dump(np.array(self.replay_buffer), file=fd)

    def reset(self):
        self.last_observation = self.env.reset()

    def plot(self, **kwargs_plot):
        self.env.render()
