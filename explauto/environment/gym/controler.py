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

    def predict(state):
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

    def predict(self, state):
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
