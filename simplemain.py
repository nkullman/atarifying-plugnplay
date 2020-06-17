import json
from sys import argv

import numpy as np

from atarifying.games import vrpssr
from atarifying import utils

def my_main(env_config):
    
    env = vrpssr.VrpssrEnv(env_config)
    s = env.reset()
    assert s in env.observation_space
    
    utils.make_image(env.render(), filename="test_i.png")

    terminal = False
    while not terminal:
        a = np.random.choice(np.arange(5))
        s, r, terminal, _ = env.step(a)
        assert s in env.observation_space
    
    utils.make_image(env.render(), filename="test_f.png")
    print(env.get_episode_summary())

if __name__ == "__main__":
    
    if len(argv) > 2:
        raise ValueError("Program takes one argument: a JSON-like environment configuration")
    
    env_config = {} if len(argv) < 2 else json.loads(argv[1])
    
    my_main(env_config)
