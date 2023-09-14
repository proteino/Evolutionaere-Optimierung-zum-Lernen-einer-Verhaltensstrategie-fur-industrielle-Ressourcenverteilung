from utilities import SimpleNetwork, MultilayerNetwork, CumulatedReward
from ContainerGym.containergym.env import ContainerEnv
from stable_baselines3.common.monitor import Monitor
import sys, json, os, time
import numpy as np
import matplotlib.pyplot as plt
import cma
from pathlib import Path

def optimize_cma(config, n_hidden_layers, n_neurons_hidden_layers, step_size, episodes_per_function_call, factor_pop_size=1):
    """runs an cma optimizer on the environment and saves all results including an evaluation graph

    Args:
        config (string): name of json configuration file to build environment from
        n_hidden_layers (int): number of hidden layers
        n_neurons_hidden_layers (_type_): number of neurons from the hidden layers
        step_size (float): initial step size of the cma optimizer
        episodes_per_function_call (int): number of episodes run per function call
        factor_pop_size(int, optional): factor to multiply the default pop size of the optimizer by 
    """

    # save log files
    dirname = "results/" + config + "/" + str(time.time())
    Path(dirname).mkdir(exist_ok=True, parents=True)
    sys.stdout = custom_logger(dirname + "/log.txt")


    # initialize environment
    env = ContainerEnv.from_json(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContainerGym/containergym/configs/" + config + ".json")
        )
    
    # choose model
    if n_hidden_layers == 0:
        model = SimpleNetwork(env.n_containers, env.n_proc_units)
    else:
        model = MultilayerNetwork(env.n_containers, env.n_proc_units, n_hidden_layers, n_neurons_hidden_layers)


    # initialize fitness function
    f = CumulatedReward(env, model, episodes_per_function_call)
    n = f.problem_dim
                  

    opts = cma.CMAOptions()
    opts["tolflatfitness"] = 1e2 # needed because the first strategies sometimes produce just the overflow penalty as reward and therefore the optimizer would terminate because of flat fitness

    default_popsize = int(4 + 3 * np.log(n)) # default popsize advised by the authors of CMA-ES
    pop_size = default_popsize * factor_pop_size # increase by factor 
    opts["popsize"] = pop_size # set new pop size in settings object

    bestx = np.random.rand(n) # initialize mean of search population randomly

    es = cma.CMAEvolutionStrategy(np.zeros(n), step_size, opts)

    while not es.stop():
        samples = es.ask() # generates new samples
        fitness = [f(s) for s in samples] # evaluate fitness of candidates
        es.tell(samples, fitness) # optimizer updates its properties
        es.disp() # print current status to console
        bestx = es.result[0] # update current minimum
        if es.sigma < 0.2: # additional termination criteria, because sometimes the optimizer wouldnt stop itself because of the noisy function
            break
    es.result_pretty() # print result to console


    # save parameters
    params = {
        "config": config,
        "n_hidden_layers": n_hidden_layers,
        "n_neurons_hidden_layers": n_neurons_hidden_layers,
        "step_size": step_size,
        "episodes_per_function_call": episodes_per_function_call,
        "pop_size": pop_size,
        "best_solution": bestx.tolist()
    }

    # Serializing json
    json_object = json.dumps(params, indent=4)
    
    # Writing to sample.json
    with open(dirname + "/params.json", "w") as file:
        file.write(json_object)
    
    generate_eval_plot(dirname)
    sys.stdout=sys.__stdout__



class custom_logger():
    """
    Saves all console output to file
    """
    def __init__(self, filename, file_header = None):
        self.console = sys.stdout
        self.file = open(filename, "w")
        if file_header:
            self.file.write(file_header + "\n")
        
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    
    def flush(self):
        self.console.flush()
        self.file.flush()


def generate_eval_plot(dirname, render_episode=False, episode_length=600):
    """_summary_

    Args:
        dirname (string): directory name
        render_episode (bool, optional): if toggled, render the episode during evaluation. Defaults to False. Does not Work currently
        episode_length (int, optional): total number of timesteps of the experiments (). Defaults to 600.
    """
    
    with open(dirname + "/params.json") as file:
        params = json.load(file)
    

    # initialize environment and model
    config = params["config"]
    n_hidden_layers = params["n_hidden_layers"]
    n_neurons_hidden_layers = params["n_neurons_hidden_layers"]
    x = params["best_solution"]


    # initialize environment
    env = ContainerEnv.from_json(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ContainerGym/containergym/configs/" + config + ".json")
        )
    env.max_episode_length = episode_length

    # choose model
    if n_hidden_layers == 0:
        model = SimpleNetwork(env.n_containers, env.n_proc_units)
    else:
        model = MultilayerNetwork(env.n_containers, env.n_proc_units, n_hidden_layers, n_neurons_hidden_layers)
    
    model.setWeights(x)

    env = Monitor(env)
    obs = env.reset()

     # Keep track of state variables (volumes), actions, and rewards
    volumes = []
    actions = []
    rewards = []
    proc_unit_indices = []
    step = 0
    episode_length = 0

    # Run episode and render
    while True:
        episode_length += 1
        prediction = model.predict(np.hstack((obs["Volumes"], obs["Time presses will be free"])))
        action = np.argmax(prediction)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        volumes.append(obs["Volumes"].copy())
        proc_unit_indices.append(info["proc_unit_indices"])
        # Toggle to render the episode
        if render_episode:
            print(volumes)
            env.render(volumes)
        rewards.append(reward)
        if done:
            break
        step += 1

    # Plot state variables during episode
    fig = plt.figure(figsize=(15, 10))
    env_unwrapped = env.unwrapped

    fig.suptitle(f"Inference using a cma trained agent", fontsize=16)

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title("Volume", fontsize=14)
    ax2.set_title("Action", fontsize=14)
    ax3.set_title("Reward", fontsize=14)

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.set_ylim(top=40)
    ax1.set_xlim(left=0, right=episode_length)
    ax2.set_yticks(list(range(env_unwrapped.action_space.n)))
    ax2.set_xlim(left=0, right=episode_length)
    ax3.set_ylim(bottom=-0.1, top=1.05)
    ax3.set_xlim(left=0, right=episode_length)
    plt.xlabel("Time Steps", fontsize=12)

    default_color = "#1f77b4"  # Default Matplotlib blue color
    color_code = {
        "C1-10": "#872657",  # raspberry
        "C1-20": "#0000FF",  # blue
        "C1-30": "#FFA500",  # orange
        "C1-40": "#008000",  # green
        "C1-50": "#B0E0E6",  # powderblue
        "C1-60": "#FF00FF",  # fuchsia
        "C1-70": "#800080",  # purple
        "C1-80": "#FF4500",  # orangered
        "C2-10": "#DB7093",  # palevioletred
        "C2-20": "#FF8C69",  # salmon1
        "C2-40": "#27408B",  # royalblue4
        "C2-50": "#54FF9F",  # seagreen1
        "C2-60": "#FF3E96",  # violet
        "C2-70": "#FFD700",  # gold1
        "C2-80": "#7FFF00",  # chartreuse1
        "C2-90": "#D2691E",  # chocolate
    }
    line_width = 3

    ## Plot volumes for each container
    for i in range(env_unwrapped.n_containers):
        ax1.plot(
            np.array(volumes)[:, i],
            linewidth=line_width,
            label=env_unwrapped.enabled_containers[i],
            color=color_code[env_unwrapped.enabled_containers[i]],
        )
    ax1.legend(bbox_to_anchor=(1.085, 1), loc="upper right", borderaxespad=0.0)

    ## Plot actions and rewards
    x_axis = range(episode_length)
    for i in x_axis:
        if actions[i] == 0:  # Action: "do nothing"
            ax2.scatter(i, actions[i], linewidth=line_width, color=default_color)
            ax3.scatter(
                i, rewards[i], linewidth=line_width, color=default_color, clip_on=False
            )
        else:
            ax2.scatter(
                i,
                actions[i],
                linewidth=line_width,
                color=color_code[env_unwrapped.enabled_containers[actions[i] - 1]],
                marker="^",
            )
            ax3.scatter(
                i,
                rewards[i],
                linewidth=line_width,
                color=color_code[env_unwrapped.enabled_containers[actions[i] - 1]],
                clip_on=False,
            )

    ax3.annotate(
        "Cum rew: {:.2f}".format(sum(rewards)),
        xy=(0.85, 0.9),
        xytext=(1.005, 0.9),
        xycoords="axes fraction",
        fontsize=13,
    )

    plt.subplots_adjust(hspace=0.5)

    plt.savefig(dirname + "/evaluation" + str(int(time.time())))
    

    

        