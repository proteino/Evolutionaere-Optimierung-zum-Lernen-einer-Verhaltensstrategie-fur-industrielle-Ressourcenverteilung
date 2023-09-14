from optimization_routine import optimize_cma


if __name__ == "__main__":
    # set parameters
    config = "5containers_2proc_units" # if you want to use a different configuration you can use the examples in ContainerGym/contaienrgym/configs
    n_hidden_layers = 1 # number of hidden layers of the artificial neural Network that implements the policy
    n_neurons_hidden_layers = 5 # dimension of the hidden layers
    step_size = 2 #initial step size of the cma-es optimizer
    episodes_per_function_call = 1 # number of episodes the fitness function runs to calculate the average collected reward of the policy
    factor_pop_size = 1 # Increases the population size of the optimizer by a factor. For Value one the default population size of cma-es is used

    optimize_cma(config, n_hidden_layers, n_neurons_hidden_layers, step_size, episodes_per_function_call, factor_pop_size=factor_pop_size)
