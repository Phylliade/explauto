import numpy as np


class Controler:
    def __init__(self, observation_space_dim, action_space_dim):
        """
        Abstract Controler object

        :param observation_space_dim:
        :param action_space_dim:
        """
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim

    def predict(self, state, **kwargs):
        raise(NotImplementedError)

    def set_parameters(self, parameters):
        self.parameters = parameters


class NNControler(Controler):
    def __init__(self, tanh=True, bias=True, **kwargs):
        super().__init__(**kwargs)

        self.tanh = tanh
        self.bias = bias

        if bias:
            bias_dim = 1
        else:
            bias_dim = 0

        self.parameter_space_dim = (self.observation_space_dim + bias_dim) * self.action_space_dim

    def predict(self, state, **kwargs):
        """
        Returns the action for a given state

        Keras compatible
        """
        # W has size (state_space_dim, action_space_dim)
        W = self.parameters[:(self.action_space_dim * self.observation_space_dim)].reshape((self.action_space_dim, self.observation_space_dim))

        if self.bias:
            # b has shape (action_space_dim)
            b = self.parameters[(self.action_space_dim * self.observation_space_dim):]
        else:
            b = 0

        action = np.dot(W, state) + b

        if self.tanh:
            action = np.tanh(action)

        return(action)

    def __call__(self, state):
        return(self.predict(state))


class IdentityControler(Controler):
    def predict(self, timestep, **kwargs):
        actions = self.parameters
        return(actions[timestep])


class SwingControler(Controler):
    def predict(self, state, **kwargs):
        """Controler dedicated to solve the MountainCar environment"""
        x = state[0]
        v = state[1]
        intensity = self.parameters[0]
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

        return([action])
