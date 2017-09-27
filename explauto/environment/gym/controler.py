import numpy as np


class Controler:
    def __init__(self, env, parameters_min=-1., parameters_max=+1.,):
        """
        Abstract Controler object

        :param observation_space_dim:
        :param action_space_dim:
        """
        self.observation_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.shape[0]

        # Bounds of each weight
        self.parameters_min = parameters_min
        self.parameters_max = parameters_max

    def predict(self, state, **kwargs):
        raise(NotImplementedError)

    def set_parameters(self, parameters):
        self.parameters = parameters


class NNControler(Controler):
    def __init__(self, tanh=True, bias=True, **kwargs):
        super().__init__(parameters_min=-1e5, parameters_max=+1e5, **kwargs)

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


class MLPControler(Controler):
    def __init__(self, layers, zeroed_output=0, tanh=True, **kwargs):
        super().__init__(**kwargs)

        self.tanh = tanh
        self.zeroed_output = zeroed_output

        # Dimensions of the different intermediate states (inputs and outputs) of the network
        self.states_dims = [self.observation_space_dim] + layers + [self.action_space_dim - zeroed_output]

        # A list storing the si zes of each layers
        self.layers_dims = list(zip(self.states_dims[:-1], self.states_dims[1:]))
        self.layers_sizes = [(x[0] + 1) * x[1] for x in self.layers_dims]

        # Do not forget the bias
        self.parameter_space_dim = sum(self.layers_sizes)

    def predict(self, state, **kwargs):
        """
        Returns the action for a given state

        Keras compatible
        """
        # W has size (state_space_dim, action_space_dim)
        output = state
        # Position in the parameter space list
        cursor = 0
        for layer_index, layer_dims in enumerate(self.layers_dims):
            # layer_size = self.layers_sizes[layer_index]
            W_size = (layer_dims[0] * layer_dims[1])
            W = self.parameters[cursor:(cursor + W_size)].reshape((layer_dims[1], layer_dims[0]))
            cursor += W_size

            b_size = 1 * layer_dims[1]
            b = self.parameters[cursor:(cursor + b_size)]
            cursor += b_size

            output = np.dot(W, output) + b

            if self.tanh:
                output = np.tanh(output)

        if self.zeroed_output != 0:
            return_value = np.zeros((self.action_space_dim, 1))
            return_value[:self.zeroed_output] = output
        else:
            return_value = output
        return(output)

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
