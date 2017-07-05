from ..environment import Environment
import gym as openai_gym
import numpy as np


class GymEnvironment(Environment):
    def __init__(self, name, observation_function=None, s_mins=None, s_maxs=None, controler="NN_unbiased"):
        env = openai_gym.make(name)

        # env = openai_gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1')

        self.env = env
        self.env.seed(123)
        self.last_observation = self.env.reset()
        param_min = -4
        param_max = 4

        if len(self.env.observation_space.shape) != 1 or len(self.env.action_space.shape) != 1:
            raise(ValueError("The action or observation space have more than one dimensions, which is not currently supported"))
        self.action_space_dim = self.env.action_space.shape[0]

        self.observation_space_dim = self.env.observation_space.shape[0]

        if controler == "NN" or controler == "NN_unbiased":
            if controler == "NN":
                self.controler = self.NN_controler
                parameter_space_dim = (self.observation_space_dim + 1) * self.action_space_dim
            elif controler == "NN_unbiased":
                self.controler = self.NN_unbiaised_controler
                parameter_space_dim = (self.observation_space_dim) * self.action_space_dim
            m_mins = [param_min] * parameter_space_dim
            m_maxs = [param_max] * parameter_space_dim
        elif controler == "Id":
            self.controler = self.Id_controler
            m_mins = np.tile(self.env.action_space.low, self.rollout_size)
            m_maxs = np.tile(self.env.action_space.high, self.rollout_size)
        elif controler == "swing":
            self.controler = self.swing_controler
            m_mins = [-1]
            m_maxs = [1]

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

    def NN_unbiaised_controler(self, state, params=None, **kwargs):
        W = params.reshape((self.action_space_dim, self.observation_space_dim))
        return(np.dot(W, state).squeeze())

    def NN_controler(self, state, params=None, **kwargs):
        """Returns the action for a given state"""
        # W has size (state_space_dim, action_space_dim)
        W = params[:(self.action_space_dim * self.observation_space_dim)].reshape((self.action_space_dim, self.observation_space_dim))

        # b has shape (action_space_dim)
        b = params[(self.action_space_dim * self.observation_space_dim):]
        return(np.dot(W, state).squeeze() + b)

    def Id_controler(self, timestep, params=None, **kwargs):
        actions = params
        return(actions[timestep])

    def swing_controler(self, state, params, **kwargs):
        """Controler dedicated to solve the MountainCar environment"""
        x = state[0]
        v = state[1]
        intensity = param[0]
        center = -0.523
        x -= center
        action = 0

        if abs(x) <= 0.1 and abs(v) <= 0.1:
            action = 0
        if x >= 0 and v >= 0.1:
            action = 1
        if x >= 0 and v < 0.01:
            action = -1
        if x < 0 and v <= -0.1:
            action = -1
        if x < 0 and v > -0.1:
            action = 1

        action *= intensity

        return(action)

    def compute_motor_command(self, actions):
        if len(actions) != self.rollout_size:
            raise(ValueError("The size of actions ({}) does not match the rollout_size ({})").format(len(actions), self.rollout_size))

        return(actions)

    def compute_sensori_effect(self, controler_parameters):
        self.last_observation = self.env.reset()

        # We store a rallout as a series of states
        # Fill the rollout with the initial position, in case the episode stops before the end
        rollout = np.tile(self.last_observation, (self.rollout_size, 1))

        for step in range(1, self.rollout_size):
            action = self.controler(self.last_observation, params=controler_parameters, timestep=step)
            observation, reward, done, info = self.env.step([action])
            rollout[step, :] = observation.squeeze()

            self.replay_buffer.append([self.last_observation, action, reward, observation])

            self.last_observation = observation

            if False:
                self.env.render()

            if done:
                if False:
                    print("Done!")
                    print("Number of steps: {}".format(step))
                break

        observation = self.observation_function(rollout)
        return(observation)

    def reset(self):
        self.last_observation = self.env.reset()

    def plot(self, **kwargs_plot):
        self.env.render()
