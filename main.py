from optimization_routine import optimize_cma


if __name__ == "__main__":
    # set parameters
    config = "5containers_2proc_units"
    n_hidden_layers = 1
    n_neurons_hidden_layers = 5
    step_size = 2 #initial step size
    episodes_per_function_call = 1

    optimize_cma(config, n_hidden_layers, n_neurons_hidden_layers, step_size, episodes_per_function_call)
