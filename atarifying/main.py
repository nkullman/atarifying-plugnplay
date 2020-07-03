import argparse
import json

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from atarifying import utils

def run(game, agent_type, env_config, total_training_steps, user_ray_config, upload_dir):
    
    ray.init(
        num_cpus=32,
        num_gpus=2 # also specify memory?
        )

    # set out the agent/trainer configurations
    agent_config = utils.get_config(agent_type)         # default agent config
    # agent_config['log_level'] = 'INFO'                # TODO make a CLI for this
    agent_config['seed'] = env_config.get('seed', None) # add seed if supplied in env_config
    agent_config['env'] = utils.get_game_env_name(game) # set the name of the environment
    agent_config['env_config'] = env_config             # set the env_config per CLI arg
    
    # we're going to search over state_types to determine which work best (using all other configs default)
    agent_config['env_config']['state_type'] = tune.grid_search(['classic', 'feature_layers', 'humangray'])

    # if we're doing the humangray state_type, then try both 1 and 4 prev frames
    agent_config['env_config']['n_frames'] = tune.sample_from(lambda spec: np.random.choice([1,4] if spec.config.state_type == 'humangray' else [1]))
    
    # if we're doing humangray, then we have to specify the convolutional layers we want to use
    # (the others, for better or worse, get flattened into one long input and get some default FCNet)
    if 'model' not in agent_config:
        agent_config['model'] = {}
    agent_config['model']['conv_filters'] = tune.sample_from(lambda spec: np.random.choice(
        [[[16,[4,4],2], [32,[4,4],2], [256,[8,8],1]]] if spec.config.state_type == 'humangray' 
        else [None]))
    
    # hard code some resource-related params based on how we're running SLURM these days
    agent_config['num_gpus'] = 1
    agent_config['num_workers'] = 16
    agent_config['num_cpus_per_worker'] = 1

    tune.run(
        agent_type, # what kind of agent to train
        name=agent_type, # name of our experiment is the name of the agent we're doing
        num_samples=5, # total number of sweeps of the state_type grid to do
        scheduler=ASHAScheduler(metric="episode_reward_mean", mode="max"), # ASHA aggressively kills bad trials
        config=agent_config,
        resources_per_trial={ # config for 2 simultaneous trials, where each trial gets 1 GPU and 16 CPUs
            "cpu":16,
            "gpu":1
        },
        checkpoint_freq=3, # Take checkpoints every 3 training iterations...
        checkpoint_at_end=True, # (also checkpoint at the end),...
        keep_checkpoints_num=1, # and only keep the best two checkpoints...
        checkpoint_score_attr='episode_reward_mean', # as determined by the mean episode reward.
        upload_dir=upload_dir # when we're done, move the results here
    )

def _get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", help="Game to play", type=str, choices=['vrpssr'], default='vrpssr')
    parser.add_argument("--agent", "-a", help="Agent type to use", type=str, choices=['APEX','DQN','DQN_RAINBOW','IMPALA','PPO'], default='DQN')
    parser.add_argument("--envconfig", "-e", help="JSON-like environment configuration", type=str, default='{}')
    parser.add_argument("--trainsteps", "-t", help="Number of steps over which to train agent", type=int, default=5e6)
    parser.add_argument("--rayconfig", "-r", help="JSON-like configuration settings for ray", type=str, default='{}')
    parser.add_argument("--upload-dir", "-u", help="Directory where results should be uploaded", type=str, default='')

    args = parser.parse_args()

    return args

def main():
    
    args = _get_args()
    run(args.game, args.agent, json.loads(args.envconfig), args.trainsteps, json.loads(args.rayconfig), args.upload_dir if args.upload_dir else None) 
