import numpy as np
import os, sys
from containergym.env import ContainerEnv
from stable_baselines3.common.monitor import Monitor


class ContainerGymRandomNetwork:
    def __init__(self, env, num_hidden_layers = 1, hidden_layer_units = [20], xavier = True):
        # xavier parameter
        self.xavier_param = lambda input_neurons, output_neurons: (np.sqrt(6 / (input_neurons + output_neurons)))

        # assert the parameters match
        if num_hidden_layers < 1 or num_hidden_layers != len(hidden_layer_units):
            raise ValueError("Number of hidden layers must match array size of hidden layer units")

        self.n_hidden_layers = num_hidden_layers

        # input dimension: the network takes each state variable as an input
        self.input_dim = env.n_containers + env.n_proc_units

        # ouput dimension: one-hot-encoding of the action space (no action included)
        self.output_dim = env.n_proc_units + 1

        # weight initialization
        if xavier:
            # input - hidden
            xavier_param_input_hidden = self.xavier_param(self.input_dim, hidden_layer_units[0])
            self.weights_input_hidden = np.random.uniform(-xavier_param_input_hidden, xavier_param_input_hidden, (self.input_dim + 1, hidden_layer_units[0]))

            # hidden - hidden
            if self.n_hidden_layers > 1:
                weights_hidden_hidden = []
                for i in range(self.n_hidden_layers - 1):
                    xavier_param_hidden_hidden = self.xavier_param(hidden_layer_units[i], hidden_layer_units[i + 1])
                    weights_hidden_hidden.append(np.random.uniform(-xavier_param_hidden_hidden, xavier_param_hidden_hidden, (hidden_layer_units[i] + 1, hidden_layer_units[i + 1])))
                self.weights_hidden_hidden = np.array(weights_hidden_hidden)
            else:
                self.weights_hidden_hidden = np.empty(0)

            # hidden output
            xavier_param_hidden_output = self.xavier_param(hidden_layer_units[-1], self.output_dim)
            self.weights_hidden_output = np.random.uniform(-xavier_param_hidden_output, xavier_param_hidden_output, (hidden_layer_units[-1] + 1, self.output_dim))
        else:
            # initialize weights with random numbers
            self.weights_input_hidden = np.random.rand(self.input_dim + 1, hidden_layer_units[0])

            if self.n_hidden_layers > 1:
                weights_hidden_hidden = []
                for i in range(self.n_hidden_layers - 1):
                    weights_hidden_hidden.append(np.random.rand(hidden_layer_units[i] + 1, hidden_layer_units[i + 1]))
                self.weights_hidden_hidden = np.array(weights_hidden_hidden)
            else:
                self.weights_hidden_hidden = np.empty(0)

            self.weights_hidden_output = np.random.rand(hidden_layer_units[-1] + 1, self.output_dim)

        # activation functions
        self.softmax = lambda x: (np.exp(x) / np.sum(np.exp(x)))
        self.logistic = lambda x: (1 / (1 + np.exp(-x)))

    

    def predict(self, obs):
        # get input data from observation
        container_volumes = obs["Volumes"]
        proc_unit_times = obs["Time presses will be free"]


        # assert the environment matches the network
        if self.output_dim != len(proc_unit_times) + 1 or self.input_dim != len(container_volumes) + len(proc_unit_times):
            raise ValueError("provided observation does not match the environment used for initialization of the network")
        

        # concatenate input data and add bias term
        input = np.hstack((container_volumes, proc_unit_times)) # concatenate state variables
        input = np.hstack(([1], input)) # add bias term


        # calulate forward pass
        
        # input layer
        output = input @ self.weights_input_hidden
        output = self.logistic(output)

        # hidden layers 
        for weights in self.weights_hidden_hidden:
            output = np.hstack(([1], output)) # add bias term
            output = output @ weights
            output = self.logistic(output)

        # output layer
        output = np.hstack(([1], output)) # add bias term
        output = output @ self.weights_hidden_output
        # output = self.softmax(output) not needed, since softmax is monoton
        
        return np.argmax(output)
        # alternative sample from multivariate distribution

        

# set environment from the config files
config_file = "5containers_2proc_units.json"

env = ContainerEnv.from_json(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "containergym/configs/" + config_file
        )
    )


for _ in range(10):
    # initialize network
    model = ContainerGymRandomNetwork(env, num_hidden_layers=1, hidden_layer_units=[3])


    # wrap and reset environment
    env = Monitor(env)
    observation = env.reset()

    # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    free_proc_units = []
    step = 0
    episode_length = 0

    # Run episode
    while True:
        episode_length += 1
        action = model.predict(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        free_proc_units.append(len(np.where(observation["Time presses will be free"] == 0)[0]))
        if done:
            break
        step += 1

    print(actions)
    print(rewards)
    print(np.sum(rewards))
    print("============================================")
















# methode für initialisierung mit flat vektor, übertragung in die matrizen
# lmmaes (oder es)
# funktion mit weight vektor als input und (negativem) kumulierten reward als output (fitness)


# fragen
# enthält flat weights vektor auch bias term? -> ich denke ja










