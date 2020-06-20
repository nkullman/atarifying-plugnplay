
# TODO do we want to add periodic re-evaluation by default?

APEX_CONFIG = {
    
    # Using pretty much the default tuned config for APE-X Atari
    # (see: https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/dqn/atari-apex.yaml)
    
    # "framework": "tf",
    "double_q": False,
    "dueling": False,
    "num_atoms": 1,
    "noisy": False,
    "n_step": 3,
    "lr": .0001,
    "adam_epsilon": .00015,
    "hiddens": [512],
    "buffer_size": 1000000,
    "exploration_config":{
        "final_epsilon": 0.01,
        "epsilon_timesteps": 200000
    },
    "prioritized_replay_alpha": 0.5,
    "final_prioritized_replay_beta": 1.0,
    "prioritized_replay_beta_annealing_timesteps": 2000000,
    "num_gpus": 1,
    
    # APEX
    "num_workers": 16,                                          # DIFF
    "num_envs_per_worker": 8,
    # "rollout_fragment_length": 20,
    "sample_batch_size": 20,
    "train_batch_size": 512,
    "target_network_update_freq": 50000,
    "timesteps_per_iteration": 25000,

    # Params to consider tweaking:
    #     items noted in main_dqn (since apex config starts from dqn config), plus... (values below are defaults)
    #     "n_step": 3,
    #     "num_gpus": 1,
    #     "num_workers": 32,
    #     "buffer_size": 2000000,
    #     "learning_starts": 50000,
    #     "train_batch_size": 512,
    #     "rollout_fragment_length": 50,
    #     "target_network_update_freq": 500000,
    #     "timesteps_per_iteration": 25000,

}

DQN_DISTRL_CONFIG = {
    
    # Using pretty much the default tuned config for Distributional RL for Atari
    # (see: https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/dqn/atari-dist-dqn.yaml)
    
    "double_q": False,
    "dueling": False,
    "num_atoms": 51,
    "noisy": False,
    "prioritized_replay": False,
    "n_step": 1,
    "target_network_update_freq": 8000,
    "lr": .0000625,
    "adam_epsilon": .00015,
    "hiddens": [512],
    "learning_starts": 20000,
    "buffer_size": 1000000,
    # "rollout_fragment_length": 4,
    "sample_batch_size": 4,
    "train_batch_size": 32,
    "exploration_config":{
        "epsilon_timesteps": 200000,
        "final_epsilon": 0.01,
    },
    "prioritized_replay_alpha": 0.5,
    "final_prioritized_replay_beta": 1.0,
    "prioritized_replay_beta_annealing_timesteps": 2000000,
    "num_gpus": 1,                                              # DIFF
    "timesteps_per_iteration": 10000,

    # Params to consider tweaking:
    #
    # "hiddens": [256], # i think we used 512 before? grid search?
    # "n_step": 1, # gridsearch [1,3]?
    # "exploration_config": {
    #         "type": "EpsilonGreedy",
    #         "initial_epsilon": 1.0,
    #         "final_epsilon": 0.02,
    #         "epsilon_timesteps": 10000,
    # },
    # "timesteps_per_iteration": 1000, # WHEN YOU CALL .train THIS IS HOW MANY STEPS ITS GOING TO DO -- NEED TO BUMP THIS UP
    # "target_network_update_freq": 500,
    # # === Replay buffer ===
    # "buffer_size": 50000,
    # "prioritized_replay_alpha": 0.6,# Alpha parameter for prioritized replay buffer.
    # "prioritized_replay_beta": 0.4,# Beta parameter for sampling from prioritized replay buffer.
    # "final_prioritized_replay_beta": 0.4,# Final value of beta (by default, we use constant beta=0.4).
    # "prioritized_replay_beta_annealing_timesteps": 20000,
    # "model": {
    #         # === For details, see https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py === #
    #         # "conv_filters": None,   # Filter config. List of [out_channels, kernel, stride] for each filter
    #         # "conv_activation": "relu",  # Nonlinearity for built-in convnet
    #         # "fcnet_activation": "tanh", # Nonlinearity for fully connected net (tanh, relu)
    #         # "fcnet_hiddens": [256, 256],    # Number of hidden layers for fully connected net
    #         # Interesteing, can toggle LSTM (options are described at above link):
    #         # "use_lstm": False,  # Whether to wrap the model with an LSTM.
    #     },
    # "optimizer": {
    #     # varies by optimizer... options?
    # },
    # "lr": 5e-4,   # The default learning rate.
    # "gamma": 0.99,  # Discount factor of the MDP.
    # "adam_epsilon": 1e-8,# Adam epsilon hyper parameter
    # "grad_clip": 40,# If not None, clip gradients during optimization at this value
    # "learning_starts": 1000,# How many steps of the model to sample before learning starts.
    # "rollout_fragment_length": 4,# Update the replay buffer with this many samples at once.
    # "train_batch_size": 32,# Size of a batch sampled from replay buffer for training.

}

DQN_RAINBOW_CONFIG = {
    
    # Using pretty much the default tuned config for Pong with Rainbow
    # (see: https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/dqn/pong-rainbow.yaml)
    
    "num_atoms": 51,
    "noisy": True,
    "gamma": 0.99,
    "lr": .0001,
    "hiddens": [512],
    "learning_starts": 10000,
    "buffer_size": 50000,
    # "rollout_fragment_length": 4,
    "sample_batch_size": 4,
    "train_batch_size": 32,
    "exploration_config":{
        "epsilon_timesteps": 2,
        "final_epsilon": 0.0
    },
    "target_network_update_freq": 500,
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.5,
    "final_prioritized_replay_beta": 1.0,
    "prioritized_replay_beta_annealing_timesteps": 400000,
    "n_step": 3,
    "gpu": True,
    "model":{
        "grayscale": True,
        "zero_mean": False,
        "dim": 42
    },
    "timesteps_per_iteration": 1000,                            # NOT DIFF this is default val; adding here for use in computing num iters

    # Params to consider tweaking:
    #
    # "hiddens": [256], # i think we used 512 before? grid search?
    # "n_step": 1, # gridsearch [1,3]?
    # "exploration_config": {
    #         "type": "EpsilonGreedy",
    #         "initial_epsilon": 1.0,
    #         "final_epsilon": 0.02,
    #         "epsilon_timesteps": 10000,
    # },
    # "timesteps_per_iteration": 1000, # WHEN YOU CALL .train THIS IS HOW MANY STEPS ITS GOING TO DO -- NEED TO BUMP THIS UP
    # "target_network_update_freq": 500,
    # # === Replay buffer ===
    # "buffer_size": 50000,
    # "prioritized_replay_alpha": 0.6,# Alpha parameter for prioritized replay buffer.
    # "prioritized_replay_beta": 0.4,# Beta parameter for sampling from prioritized replay buffer.
    # "final_prioritized_replay_beta": 0.4,# Final value of beta (by default, we use constant beta=0.4).
    # "prioritized_replay_beta_annealing_timesteps": 20000,
    # "model": {
    #         # === For details, see https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py === #
    #         # "conv_filters": None,   # Filter config. List of [out_channels, kernel, stride] for each filter
    #         # "conv_activation": "relu",  # Nonlinearity for built-in convnet
    #         # "fcnet_activation": "tanh", # Nonlinearity for fully connected net (tanh, relu)
    #         # "fcnet_hiddens": [256, 256],    # Number of hidden layers for fully connected net
    #         # Interesteing, can toggle LSTM (options are described at above link):
    #         # "use_lstm": False,  # Whether to wrap the model with an LSTM.
    #     },
    # "optimizer": {
    #     # varies by optimizer... options?
    # },
    # "lr": 5e-4,   # The default learning rate.
    # "gamma": 0.99,  # Discount factor of the MDP.
    # "adam_epsilon": 1e-8,# Adam epsilon hyper parameter
    # "grad_clip": 40,# If not None, clip gradients during optimization at this value
    # "learning_starts": 1000,# How many steps of the model to sample before learning starts.
    # "rollout_fragment_length": 4,# Update the replay buffer with this many samples at once.
    # "train_batch_size": 32,# Size of a batch sampled from replay buffer for training.
}

IMPALA_CONFIG = {
    
    # Using pretty much the default tuned config for IMPALA for Atari
    # (see: https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/impala/atari-impala.yaml)
    
    # "rollout_fragment_length": 50,
    "sample_batch_size": 50,
    "train_batch_size": 500,
    "num_workers": 16,
    "num_envs_per_worker": 5,
    "clip_rewards": True,
    "lr_schedule": [
        [0, 0.0005],
        [20000000, 0.000000000001],
    ],
}

PPO_CONFIG = {
    
    # Using pretty much the default tuned config for PPO Atari
    # (see: https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/atari-ppo.yaml)
    
    # "framework": "tf",
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_rewards": True,
    "clip_param": 0.1,
    "vf_clip_param": 10.0,
    "entropy_coeff": 0.01,
    "train_batch_size": 5000,
    # "rollout_fragment_length": 100,
    "sample_batch_size": 100,
    "sgd_minibatch_size": 500,
    "num_sgd_iter": 10,
    "num_workers": 16,                                          # DIFF
    "num_envs_per_worker": 5,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "vf_share_layers": True,
    "num_gpus": 1,

    # Params to consider tweaking:
    #     # Clip param for the value function. Note that this is sensitive to the
    #     # scale of the rewards. If your expected V is large, increase this.
    #     "vf_clip_param": 10.0,
    #     "rollout_fragment_length": 200,
    #     # Number of timesteps collected for each SGD round. This defines the size
    #     # of each SGD epoch.
    #     "train_batch_size": 4000,
    #     # Total SGD batch size across all devices for SGD. This defines the
    #     # minibatch size within each epoch.
    #     "sgd_minibatch_size": 128,

}

_configs = {
    'APEX': APEX_CONFIG,
    'DQN': DQN_DISTRL_CONFIG,
    'DQN_RAINBOW': DQN_RAINBOW_CONFIG,
    'IMPALA': IMPALA_CONFIG,
    'PPO': PPO_CONFIG,
}

_steps_per_train_iter = {
    'APEX': APEX_CONFIG["timesteps_per_iteration"],
    'DQN': DQN_DISTRL_CONFIG["timesteps_per_iteration"],
    'DQN_RAINBOW': DQN_RAINBOW_CONFIG["timesteps_per_iteration"],
    'IMPALA': IMPALA_CONFIG["train_batch_size"],
    'PPO': PPO_CONFIG["train_batch_size"] * PPO_CONFIG['num_sgd_iter'],
}

def get_config(agent_type:str):
    return _configs.get(agent_type.upper(), None)

def get_steps_per_training_iter(agent_type:str):
    return _steps_per_train_iter.get(agent_type.upper(), None)
