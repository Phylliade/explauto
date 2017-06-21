from ..environment import Environment
import gym as openai_gym
import numpy as np


class GymEnvironment(Environment):
    def __init__(self, name):
        env = openai_gym.make(name)

        #env = openai_gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        self.env = env
        self.env.seed(123)
        self.last_observation = self.env.reset()

        self.rollout_size = 1000

        self.replay_buffer = []

        if len(self.env.observation_space.shape) != 1 or len(self.env.action_space.shape) != 1:
            raise(ValueError("The action or observation space have more than one dimensions, which is not currently supported"))
        self.action_space_dim = self.env.action_space.shape[0]
        self.observation_space_dim = self.env.observation_space.shape[0]

        Environment.__init__(
            self,
            np.tile(self.env.action_space.low, self.rollout_size),
            np.tile(self.env.action_space.high, self.rollout_size),
            np.tile(self.env.observation_space.low, self.rollout_size),
            np.tile(self.env.observation_space.high, self.rollout_size)
        )

    def compute_motor_command(self, actions):
        if len(actions) != self.rollout_size:
            raise(ValueError("The size of actions ({}) does not match the rollout_size ({})").format(len(actions), self.rollout_size))
        return(actions)

    def compute_sensori_effect(self, actions):
        # We store a rallout as a series of states
        rollout = np.zeros((self.rollout_size, self.observation_space_dim))

        for step, action in enumerate(actions):
            observation, reward, done, info = self.env.step([action])
            rollout[step, :] = observation
            self.replay_buffer.append([self.last_observation, action, reward, observation])

            self.last_observation = observation

            if done:
                print("Done!")
                break

        return(rollout.flatten())

    def reset(self):
        self.env.reset()

    def plot(self, **kwargs_plot):
        self.env.render()