import argparse
import json
import logging
import math

import ray
from ray.tune.logger import pretty_print

from atarifying import utils

def run(game, agent_type, env_config, total_training_steps, user_ray_config):
    
    # initialize ray
    ray_config = utils.get_ray_config(game, agent_type) # default ray config
    for k,v in user_ray_config.items():
        ray_config[k] = v                               # overwritten/supplemented by user-specified config
    ray.init(**ray_config)
    
    # set out the agent/trainer configurations
    agent_config = utils.get_config(agent_type)         # default agent config
    agent_config['seed'] = env_config.get('seed', None) # add seed if supplied in env_config
    agent_config['env_config'] = env_config             # set the env_config per CLI arg
    agent_config['log_level'] = 'INFO'                  # TODO make a CLI for this

    # make changes according to the specified game/agent/environment
    agent_config_mods = utils.get_agent_config_mods(game, agent_type, env_config)
    for k,v in agent_config_mods.items():
        agent_config[k] = v

    num_training_iters = math.ceil(total_training_steps / utils.get_steps_per_training_iter(agent_type))   # how many training iterations to perform
    checkpoint_every = math.floor(0.1*num_training_iters) # save checkpoints approx every 10% of completed training
    
    trainer = utils.get_trainer(agent_type)(env=utils.get_game_env(game), config=agent_config)

    for training_iteration in range(num_training_iters):
        result = trainer.train()
        logging.info(pretty_print(result))

        if training_iteration % checkpoint_every == 0 or training_iteration == num_training_iters-1:
            checkpoint = trainer.save()
            logging.info("checkpoint saved at", checkpoint)
    
    logging.info("Training complete")

    # OPTION 2: Could run with tune: "from ray import tune"
    # agent_config['env'] = "Vrpssr-v0" # TODO in utils, make a get_game_env_name(game)
    # tune.run(
    #     agent_type,
    #     stop={"episode_reward_mean": 200}, # can specify a time limit as well. maybe something like (if mean rwd sucks after X iters, then quit?)
    #     config=agent_config, # Nice thing is that you can easily request hparam tuning in the config: tune.grid_search([0.01, 0.001, 0.0001]),
    #     # Also set checkpoint freq and checkpoint at end
    # )

def _get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", help="Game to play", type=str, choices=['vrpssr'], default='vrpssr')
    parser.add_argument("--agent", "-a", help="Agent type to use", type=str, choices=['APEX','DQN','DQN_RAINBOW','IMPALA','PPO'], default='DQN')
    parser.add_argument("--envconfig", "-e", help="JSON-like environment configuration", type=str, default='{}')
    parser.add_argument("--trainsteps", "-t", help="Number of steps over which to train agent", type=int, default=5e6)
    parser.add_argument("--rayconfig", "-r", help="JSON-like configuration settings for ray", type=str, default='{}')

    args = parser.parse_args()

    return args

def main():
    
    args = _get_args()
    run(args.game, args.agent, json.loads(args.envconfig), args.trainsteps, json.loads(args.rayconfig)) 
