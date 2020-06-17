import gym
from matplotlib import pyplot as plt
import numpy as np
from ray.rllib import agents

from atarifying import games
from atarifying import configs

def rbg_to_gray(rgb):
    """Convert an RGB image to a grayscale one"""
    assert rgb.shape[0] == 3, "RGB array must be channels-first"
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def dist(p1, p2, axis=None, lnorm=1):
    """Returns the distance as measured by the lnorm provided.
    Defaults to lnorm=1 (Manhattan distance).
    """
    return np.linalg.norm(np.array(p1)-np.array(p2), axis=axis, ord=lnorm)

def make_image(pixel_array;np.array, filename:str=None) -> None:
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
        return agents.dqn.ImpalaTrainer
    
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
        return games.vrpssr.VrpssrEnv
    else:
        return None
