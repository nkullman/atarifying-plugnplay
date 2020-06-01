import numpy as np

import environment
import utils

def main():
    
    env = environment.VrpssrEnv({'state_type':'classic'})
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
    main()