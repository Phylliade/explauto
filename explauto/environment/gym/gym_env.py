from ..environment import Environment
from .controler import NNControler, MLPControler
import gym as openai_gym
import scipy.stats
import numpy as np
import pickle


DEBUG = False


class GymEnvironment(Environment):
    def __init__(self, env, controler, observation_function=None, s_mins=None, s_maxs=None, hooks=None):
        # env = openai_gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        self.env = env
        self.env.seed(123)
        self.last_observation = self.env.reset()

        if len(self.env.observation_space.shape) != 1 or len(self.env.action_space.shape) != 1:
            raise(ValueError("The action or observation space have more than one dimensions, which is not currently supported"))
        self.action_space_dim = self.env.action_space.shape[0]

        self.observation_space_dim = self.env.observation_space.shape[0]

        if controler == "NN":
            self.controler = NNControler(tanh=False, bias=True, env=env)
        if controler == "NN_tanh":
            self.controler = NNControler(tanh=True, bias=True, env=env)
        elif controler == "NN_unbiased":
            self.controler = NNControler(tanh=False, bias=False, env=env)
        elif controler == "NN_tanh_unbiased":
            self.controler = NNControler(bias=False, tanh=True, env=env)
        elif controler == "MLP":
            self.controler = MLPControler(observation_space_dim=self.observation_space_dim, action_space_dim=self.action_space_dim)
        else:
            self.controler = controler

        m_mins = [self.controler.parameters_min] * self.controler.parameter_space_dim
        m_maxs = [self.controler.parameters_max] * self.controler.parameter_space_dim

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

    def compute_sensori_effect(self, controler_parameters, render=False, save_to_replay_buffer=False, noisy_action=False, noise_intensity=1.0, hooks=None):
        # Start a new episode
        self.last_observation = self.env.reset()

        episode_reward = 0

        # We store a rallout as a series of states
        # Fill the rollout with the initial position, in case the episode stops before the end
        rollout = np.tile(self.last_observation, (self.rollout_size, 1))

        for step in range(1, self.rollout_size):
            self.controler.set_parameters(controler_parameters)
            action = self.controler(self.last_observation)
            if noisy_action:
                # TODO: Use a truncated normal instead, to avoid clipping
                # action += np.random.normal(scale=noise_intensity, size=self.action_space_dim)
                action += scipy.stats.truncnorm.rvs(-1 - action, 1 - action, scale=noise_intensity, size=self.action_space_dim)
            # Clip the action in the bounds defined in self.env
            # action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            observation, reward, done, info = self.env.step(action)
            rollout[step, :] = observation.squeeze()

            episode_reward += reward

            terminal = (step == (self.rollout_size - 1)) or done

            # Save to replay buffer
            if save_to_replay_buffer:
                # TODO: Use rl's memory object instead of a numpy array
                self.replay_buffer.append([self.last_observation, action, reward, observation, terminal])

            self.last_observation = observation

            if hooks is not None:
                hooks.step_end()

            if render:
                self.env.render()

            if terminal:
                if DEBUG:
                    print("Done!")
                    print("Number of steps: {}".format(step))
                break

        observation = self.observation_function(rollout)
        return((observation, episode_reward))

    def save_replay_buffer(self, file="replay_buffer.p"):
        print("Saving replay buffer")
        with open(file, "wb") as fd:
            pickle.dump(np.array(self.replay_buffer), file=fd)

    def dump_replay_buffer(self):
        return(self.replay_buffer)

    def reset(self):
        self.last_observation = self.env.reset()

    def reset_replay_buffer(self):
        self.replay_buffer = []

    def plot(self, **kwargs_plot):
        self.env.render()
