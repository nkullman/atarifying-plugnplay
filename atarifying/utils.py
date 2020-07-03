import gym
from matplotlib import pyplot as plt
import numpy as np
from ray.rllib import agents

from atarifying import configs
from atarifying.games import vrpssr

def rbg_to_gray(rgb):
    """Convert an RGB image to a grayscale one"""
    assert rgb.shape[0] == 3, "RGB array must be channels-first"
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def dist(p1, p2, axis=None, lnorm=1):
    """Returns the distance as measured by the lnorm provided.
    Defaults to lnorm=1 (Manhattan distance).
    """
    return np.linalg.norm(np.array(p1)-np.array(p2), axis=axis, ord=lnorm)

def make_image(pixel_array:np.array, filename:str=None) -> None:
    """Generate the image from the pixel array

    Args:
        pixel_array (np.array): pixel array to make into an image
        filename (str, optional): Where to save the image file. Defaults to None, which just shows the plot and does not save it.
    """
    
    plt.imshow(pixel_array, interpolation='nearest')
    plt.axis('off')
    
    # if there's only one pixel array, we assume the image is grayscale
    if pixel_array.ndim == 2:
        plt.gray()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')

def get_config(agent_type:str):
    return configs.get_config(agent_type)

def get_steps_per_training_iter(agent_type:str):
    return configs.get_steps_per_training_iter(agent_type)

def get_trainer(agent_type:str):
    agent_key = agent_type.upper()
    
    if agent_key == 'APEX':
        return agents.dqn.ApexTrainer
    
    elif agent_key == 'DQN' or agent_key == 'DQN_RAINBOW':
        return agents.dqn.DQNTrainer
    
    elif agent_key == 'IMPALA':
        return agents.impala.ImpalaTrainer
    
    elif agent_key == 'PPO':
        return agents.ppo.PPOTrainer

    else:
        return None

def get_game_env(game:str) -> gym.Env:
    """Returns the gym environment for the game.
    NB: returns the *class itself* (not an instantiation of the class)

    Args:
        game (str): the game for which to get the environment 

    Returns:
        gym.Env: the class of the gym environment for the game
    """
    if game.lower() == 'vrpssr':
        return vrpssr.VrpssrEnv
    else:
        return None

def get_game_env_name(game:str) -> str:
    """Returns the name of the gym environment for the game.

    Args:
        game (str): the game for which to get the environment 

    Returns:
        str: the name of the gym environment for the game
    """
    if game.lower() == 'vrpssr':
        return 'Vrpssr-v0'
    else:
        return None

def get_ray_config(game:str, agent_type:str) -> dict:
    """Returns the default ray config for running an `agent_type` agent on `game`.
    Used to retrieve default resource allocations (memory, num_cpus, num_gpus).

    Args:
        game: name of the game
        agent_type: name of the agent

    Returns:
        Dict (default {}): ray configs
    """
    if game.lower() == 'vrpssr':
        return vrpssr.VrpssrConfigs().CONFIGS['ray'][agent_type]
    else:
        return {}

def get_agent_config_mods(game:str, agent_type:str, env_config:dict) -> dict:
    """Returns the default ray config for running an `agent_type` agent on `game`.
    Used to retrieve default resource allocations (memory, num_cpus, num_gpus).

    Args:
        game: name of the game
        agent_type: name of the agent

    Returns:
        Dict (default {}): ray configs
    """
    mods = {}
    if game.lower() == 'vrpssr':
        configs = vrpssr.VrpssrConfigs().CONFIGS['agent']
        
        # add config settings that apply to all agents playing the game
        for k,v in configs["all"].items():
            mods[k] = v
        
        # get config settings that apply to just some agents
        if agent_type in configs:
            for k,v in configs[agent_type].items():
                mods[k] = v

        # get config settings that apply to just some `state_type`s
        if env_config and 'state_type' in env_config and env_config['state_type'] in configs:
            for k,v in configs[env_config['state_type']].items():
                mods[k] = v
    
    # Other games...
    
    return mods
