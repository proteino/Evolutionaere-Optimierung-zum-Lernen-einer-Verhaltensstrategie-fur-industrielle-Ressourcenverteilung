import numpy as np
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt


# activation functions

def softmax(x):
    return (np.exp(x) / np.sum(np.exp(x)))

def logistic(x):
    (1 / (1 + np.exp(-x)))


class SimpleNetwork:
    """Simple Network with no hidden layers. 
    The number of input neurons depends on the number of state variables(number of containers + number of processing units).
    The Output dimension depends on the action space of the environment(number of containers + 1)
    """
    def __init__(self, n_containers_env, n_proc_units_env):

        # set parameters
        self.weights_set = False
        self.input_dim = n_containers_env + n_proc_units_env # number of state variables of container gym environment
        self.output_dim = n_containers_env + 1 # action space of environment 

        self.n_weights_input_output = (self.input_dim + 1) * self.output_dim
        self.n_all_weights = self.n_weights_input_output

    
    def setWeights(self, flat_weight_vector):
        """sets the weights of the network

        Args:
            flat_weight_vector (array_like): One-dimensional array with the new weights.

        Raises:
            ValueError: Raises a ValueError, if the number of weights does not fit the network dimensions.
        """
        if len(flat_weight_vector) != self.n_all_weights:
            raise ValueError(f"Length of weight vector ({len(flat_weight_vector)}) does not match network dimensions. Network has {self.n_all_weights} weights")


        flat_weight_vector = np.array(flat_weight_vector)

        # fill and reshape weight vectors    
        self.weights_input_output = flat_weight_vector.reshape((self.input_dim + 1, self.output_dim))
        self.weights_set = True


    def predict(self, input):
        """Calculates the forward pass of the network

        Args:
            input (array_like): Array with the state variables of the container gym environment

        Raises:
            KeyError: The weights of this network need to be set with the setWeights() function before calling the predict() function. Raises a KeyError if this was not done.
            ValueError: Raises a ValueError, if the provided input does not match the input dimension of the network

        Returns:
            array_like: Returns a value for each possible action
        """
        if not self.weights_set:
            raise KeyError("weights have not been set yet")
        
        input = np.array(input)
        if len(input) != self.input_dim:
            raise ValueError(f"provided input does not match input dimension of network. Expected dimension {self.input_dim.shape} but received {input.shape}")

        # calculate forward pass
        input = np.hstack(([1], input)) # add bias term
        output = input @ self.weights_input_output
        output = softmax(output)

        return output
    

class MultilayerNetwork:
    """Neural Network with a dynamic number of hidden layers. Input and output dimension depend on the container gym environment
    """
    def __init__(self, n_containers_env, n_proc_units_env, n_hidden_layers, n_neurons_hidden_layers):
        """

        Args:
            n_containers_env (int): Number of containers in the container gym environment
            n_proc_units_env (int): Number of processing units in the container gym environment
            n_hidden_layers (int): Number of hidden layers.
            n_neurons_hidden_layers (int): Number of neurons of the hidden layers.
        """

        # set parameters
        self.weights_set = False
        self.input_dim = n_containers_env + n_proc_units_env # number of state variables of container gym environment
        self.output_dim = n_containers_env + 1 # action space of environment 

        self.hidden_dim = n_neurons_hidden_layers
        self.n_hidden_layers = n_hidden_layers

        self.n_weights_input_hidden = (self.input_dim + 1) * self.hidden_dim
        self.n_weights_hidden_hidden = (self.n_hidden_layers - 1) * (self.hidden_dim + 1) * self.hidden_dim
        self.n_weights_hidden_output = (self.hidden_dim + 1) * self.output_dim


        self.n_all_weights = self.n_weights_input_hidden + self.n_weights_hidden_output + self.n_weights_hidden_hidden

        

    
    def setWeights(self, flat_weight_vector):
        """sets the weights of the network

        Args:
            flat_weight_vector (array_like): One-dimensional array with the new weights. 

        Raises:
            ValueError: Raises a ValueError, if the number of weights does not fit the network dimensions. 
        """
        if len(flat_weight_vector) != self.n_all_weights:
            raise ValueError(f"Length of weight vector ({len(flat_weight_vector)}) does not match network dimensions. Network has {self.n_all_weights} weights")

        flat_weight_vector = np.array(flat_weight_vector)

        # fill and reshape weight vectors    
        self.weights_input_hidden = flat_weight_vector[0:self.n_weights_input_hidden].reshape((self.input_dim + 1, self.hidden_dim))

        self.weights_hidden = flat_weight_vector[self.n_weights_input_hidden:self.n_weights_input_hidden + self.n_weights_hidden_hidden].reshape((self.n_hidden_layers - 1, self.hidden_dim + 1, self.hidden_dim))

        self.weights_hidden_output = flat_weight_vector[self.n_weights_input_hidden + self.n_weights_hidden_hidden:].reshape((self.hidden_dim + 1, self.output_dim))

        self.weights_set = True


    def predict(self, input):
        """Calculates the forward pass of the network

        Args:
            input (array_like): Array with the state variables of the container gym environment

        Raises:
            KeyError: The weights of this network need to be set with the setWeights() function before calling the predict() function. Raises a KeyError if this was not done.
            ValueError: Raises a ValueError, if the provided input does not match the input dimension of the network

        Returns:
            array_like: Returns a value for each possible action
        """
        if not self.weights_set:
            raise KeyError("weights have not been set yet")
        
        input = np.array(input)
        if len(input) != self.input_dim:
            raise ValueError(f"provided input does not match input dimension of network. Expected dimension {self.input_dim.shape} but received {input.shape}")
        
        # calculate forward pass

        # input_hidden
        input = np.hstack(([1], input)) # add bias term
        output = input @ self.weights_input_hidden
        output = np.tanh(output)

        # hidden_hidden
        for i in range(self.n_hidden_layers - 1):
            output = np.hstack(([1], output))
            output = output @ self.weights_hidden[i]
            # output = self.logistic(output)
            output = np.tanh(output)

        # hidden_output
        output = np.hstack(([1], output))
        output = output @ self.weights_hidden_output
        output = softmax(output)

        return output

    
    

class CumulatedReward(object):
    def __init__(self, environment, model, episodes=1):
        """
        Args:
            environment: Container Gym Environment, which returns the reward of the predicted actions
            model: model to predict an action
            episodes (int): number of episodes to run 
        """
        if model.input_dim != environment.n_containers + environment.n_proc_units:
            raise KeyError("The models input dimension has to match the number of state variables of the environment")
        
        if model.output_dim != environment.n_containers + 1:
            raise KeyError("The models output dimension has to match the action space of the environment")
        
        if episodes < 1:
            raise ValueError("Cannot run less than one episode")
        
        self.env = environment
        self.model = model

        self.episodes = episodes
        self.problem_dim = self.model.n_all_weights

        self.performances = []
        self.best_performance = -np.inf
        self.i_best_performance = None
        self.num_function_calls = 0


    def __call__(self, flat_weight_vector):
        """Returns the negated cumulative reward of the container gym environment, with a neural network used to predict the actions based on 
        the state of the environment. When specified on initialization, runs multiple episodes and averages the reward.

        Args:
            flat_weight_vector (array_like): vector with the weights, which will be used by the model to predict the actions

        Returns:
            float: negated cumulative reward of the actions predicted by the model, possibly averaged over multiple episoded
        """
        self.num_function_calls += 1
        # fill network with parameters, predict actions and return negative cumulative reward
        self.model.setWeights(flat_weight_vector)

        # wrap environment
        env = Monitor(self.env)

        # save the cumulated rewards of all episodes
        cumulated_rewards = []
        steps = []
        actions = []
        
        for episode in range(self.episodes):
            # run episode
            observation = env.reset()
            rewards = []
            actions_per_episode = []
            steps_per_episode = 0
            while True:
                prediction = self.model.predict(np.hstack((observation["Volumes"], observation["Time presses will be free"])))
                #if np.random.rand() < 0.001:
                    # print(prediction) #print some predictions for debugging
                action = np.argmax(prediction)
                observation, reward, done, info = env.step(action)
                
                steps_per_episode += 1
                actions_per_episode.append(action)
                rewards.append(reward)
                if done:
                    break
            
            cumulated_rewards.append(np.sum(rewards))
            actions.append(actions_per_episode)
            steps.append(steps_per_episode)

        mean_cumulated_rewards = np.mean(cumulated_rewards)

        # save performance and check if best performance
        self.performances.append({"cumulated_rewards": cumulated_rewards, "actions": actions, "steps": steps,"weight_vector": flat_weight_vector})
        if mean_cumulated_rewards > self.best_performance:
            self.best_performance = mean_cumulated_rewards
            self.i_best_performance = self.num_function_calls - 1

        return -mean_cumulated_rewards
    

    def evaluate_solution(self, x = None):
        """returns a graph with an evaluation of the provided solution. If none is provided, uses the best recorded performance
        """
        if x is None:
            # get current best solution and set the model accordingly
            solution = self.performances[self.i_best_performance]["weight_vector"]
            self.model.setWeights(solution)
        else:
            self.model.setWeights(x)

        # wrap environment
        env = Monitor(self.env)

        # Keep track of state variables (volumes), actions, and rewards
        volumes = []
        actions = []
        rewards = []
        free_proc_units = []
        step = 0
        episode_length = 0

        # Run episode
        observation = env.reset()
        while True:
            episode_length += 1
            prediction = self.model.predict(np.hstack((observation["Volumes"], observation["Time presses will be free"])))
            action = np.argmax(prediction)
            actions.append(action)
            observation, reward, done, info = self.env.step(action)
            rewards.append(reward)
            free_proc_units.append(len(np.where(observation["Time presses will be free"] == 0)[0]))
            volumes.append(observation["Volumes"])
            if done:
                break
            step += 1
        


        x = np.arange(len(actions))

        rewards = np.array(rewards)
        # col = np.empty(len(actions), dtype=object)
        # col[rewards == 1] = "lime"
        # col[0 < rewards and rewards < 1] = "green"
        # col[rewards == 0] = "black"
        # col[-1 < rewards  and rewards < 0] = "tomato"
        # col[rewards == -1] = "darkred"

        col = np.where(rewards>0, "g", np.where(rewards<0, "r", "b"))

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("step")
        ax1.set_ylabel("action")
        ax1.set_ylim(([0, self.env.n_containers + 0.1]))
        ax1.scatter(x, actions, color=col, marker=".")

        # ax2 = ax1.twinx()
        # ax2.set_ylabel("reward")
        # ax2.set_ylim = [-1 , 1]
        # ax2.bar(x, rewards)

        fig.tight_layout()

        return fig