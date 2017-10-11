import numpy as np


class Controler:
    def __init__(self, env, parameter_space_dim, parameters_min=-1., parameters_max=+1.):
        """
        Abstract Controler object

        :param observation_space_dim:
        :param action_space_dim:
        """
        # Observation space
        self.observation_space_dim = env.observation_space.shape[0]
        self.observation_space_low = env.observation_space.low
        self.observation_space_high = env.observation_space.high

        # Action space
        self.action_space_dim = env.action_space.shape[0]
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high

        # Parameter space
        self.parameter_space_dim = parameter_space_dim

        # Bounds of each parameter
        self.parameters_min = parameters_min
        self.parameters_max = parameters_max

        # List of weights
        # This is a dict equivalent of the `parameters` attribute
        self.weights = {}

    def predict(self, state, **kwargs):
        raise(NotImplementedError)

    def __call__(self, state, **kwargs):
        return(self.predict(state, **kwargs))


    def set_parameters(self, parameters):
        self.parameters = parameters

    def get_weights(self):
        return self.weights


class RandomControler(Controler):
    def __init__(self, env):
        # We emulate a parameter space
        super(RandomControler, self).__init__(env=env, parameter_space_dim=1)

    def predict(self, state):
        """Choose uniformly distributed actions"""
        action = np.random.uniform(low=self.action_space_low, high=self.action_space_high, size=(self.action_space_dim,))

        return(action)


class NNControler(Controler):
    def __init__(self, tanh=True, bias=True, **kwargs):
        self.tanh = tanh
        self.bias = bias

        if bias:
            bias_dim = 1
        else:
            bias_dim = 0

        parameter_space_dim = (self.observation_space_dim + bias_dim) * self.action_space_dim

        super().__init__(parameter_space_dim=parameter_space_dim, parameters_min=-1e5, parameters_max=+1e5, **kwargs)

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

        # Save weights
        self.weights["w"] = W
        self.weights["b"] = b

        return(action)


class MLPControler(Controler):
    def __init__(self, layers, zeroed_output=0, tanh=True, **kwargs):

        self.tanh = tanh
        self.zeroed_output = zeroed_output

        # Dimensions of the different intermediate states (inputs and outputs) of the network
        self.states_dims = [self.observation_space_dim] + layers + [self.action_space_dim - zeroed_output]

        # A list storing the sizes of each layers
        self.layers_dims = list(zip(self.states_dims[:-1], self.states_dims[1:]))
        self.layers_sizes = [(x[0] + 1) * x[1] for x in self.layers_dims]

        # Do not forget the bias
        parameter_space_dim = sum(self.layers_sizes)
        super(MLPControler, self).__init__(parameter_space_dim=parameter_space_dim, **kwargs)

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
