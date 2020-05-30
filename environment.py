from collections import deque
import math

import gym
from gym import spaces
from gym.envs.registration import register
from game import Game
import numpy as np

register(
    id='VRPSSR-v0',
    entry_point='environment:VrpssrEnv'
)
register(
    id='VRPSSRTest-v0',
    entry_point='environment:VrpssrEnvTest'
)

class VrpssrEnv(gym.Env):
    metadata = {
        'render.modes': ['pixelgray','pixelrgb','binarystack']
    }

    def __init__(self):
        self.game = None
        self.type = None
        self.rwtype = None
        self.mask = False
        self.curr_state = None
        self.seed = None
        self.rng_master = None
        self.rng_inst = None
        self.initialized = False
        self.frame_stack = None
        self.nframes = None

    def init(self, rtype, reward, mask, seed=None, nframes=1):
        if reward not in ["custom", "paper", "paperlike", "linear", "piecewise", "quadratic", "reward", "mask","ndk","ndkp","ndkn"]:
            raise ValueError("Invalid RPS")
        
        self.type = rtype
        self.rwtype = reward
        self.mask = mask
        self.curr_state = None
        self.seed = seed
        self.nframes = 1 if self.type in ['binarystack','classic'] else nframes
        
        self.frame_stack = deque(maxlen=nframes) # holds the last nframes number of game renders
        
        # setup random generators
        self.rng_master = np.random.RandomState(seed=self.seed) # to spawn seeds for other RNGs
        self.rng_inst = np.random.RandomState(seed=self.rng_master.randint(np.iinfo(np.int32).max)) # for instance-related randomness
        
        self.initialized = True

    def get_pi_state(self):
        assert self.curr_state is not None, "Game must be reset in order to retrieve the init PI state"
        return self.game.get_pi_state()

    def step(self, action):
        assert self.curr_state is not None, "Game must be reset in order to step"
        (next_frame, reward, terminal, mask) = self.game.step(action)
        self.frame_stack.append(next_frame)
        self.curr_state = self._state_from_frames()
        return self.curr_state, reward, terminal, mask

    def _state_from_frames(self):
        if self.type in ['binarystack','classic']:
            return self.frame_stack[-1] # return just the most recent "frame"
        else:
            # returning a stack of frames
            state = np.array(self.frame_stack)
            # if we don't have as many frames as would be expected, then we repeat the most recent frame
            if state.shape[0] < self.nframes:
                rep_times = self.nframes - state.shape[0] # how many frames are we missing?
                state = np.pad(state, pad_width=[(0,rep_times),(0,0),(0,0)], mode='edge') # repeat the last one that many times
            return state

    def reset(self):
        if not self.initialized: 
            raise RuntimeError("Environment must first be initialized.")
        
        self.frame_stack.clear()
        
        self._generate_game()
        
        new_frame = self.game.render()
        self.frame_stack.append(new_frame)
        self.curr_state = self._state_from_frames()
        
        return self.curr_state

    def _generate_game(self):
        
        def _locate_customer(rng, size, dist_type, big=40, small=30):
            """Places a customer on the game board following the method described
            in Ulmer et al (2017).
            """
            

            if dist_type == 'random':
                return tuple(rng.randint(0, size, size=(2,)))
            
            else:
                
                cluster_means = []
                
                # cluster 1 at (10,10) -- scaled down for the smaller instances
                meanx = (big/4) * (1 if size==big else 0.75)
                meany = meanx
                cluster_means.append((meanx,meany))
                
                # cluster 2 at (10,30)
                # meanx unchanged (still at 10)
                meany = ((3/4)*big) * (1 if size==big else 0.75)
                cluster_means.append((meanx,meany))
                
                # cluster 3 at (30,20) if need be
                if dist_type == 'cluster3':
                    meanx = meany # at 30
                    meany = (big/2) * (1 if size==big else 0.75)
                    cluster_means.append((meanx,meany))

                # probability of which cluster a customer belongs to
                wts = [0.5,0.5] if dist_type == 'cluster2' else [0.25,0.5,0.25]

                # draw a cluster in which to stick the customer
                means = cluster_means[rng.choice(range(len(cluster_means)), p=wts)]

                # randomly draw x,y near the cluster
                # (not exactly like Ulmer et al, since this will have stddev sqrt(2) km, but close enough. 
                # sort of necessary anyways, since we're fishing for unique int pairs, which isn't as easy as unique float pairs)
                return int(np.rint(rng.normal(means[0],2))), int(np.rint(rng.normal(means[1],2)))
        
        # small,big = 30,40
        big = 32
        small = (3*big) // 4

        size_choices = [big] #[small,big]
        size = self.rng_inst.choice(size_choices)
        
        remaining_time = (225*(big**2)) // 1000  # set in accordance with the len/time ratio of 360min:20km = 360 min:40 grid spaces

        depot_pos = (size // 2, size // 2)
        
        lambda_choices = [15] # [25, 50, 75]
        cust_lambda = self.rng_inst.choice(lambda_choices)
        
        customer_distribution_choices = ['cluster3'] # ['random','cluster2','cluster3']
        customer_dist = self.rng_inst.choice(customer_distribution_choices)
        
        customers = []
        customer_times = []

        exp_num_cust = 30
        
        # initizlize the number of customers newly requesting service for each time interval
        num_requesting = np.zeros((remaining_time,), dtype=np.int32)
        
        num_requesting[0] = self.rng_inst.poisson(exp_num_cust - cust_lambda) # number of customers that will be requesting service initially (at t=0)
        num_requesting[1:] = self.rng_inst.poisson(cust_lambda/(remaining_time-1), size=(remaining_time-1))
        for i, num_custs in enumerate(num_requesting):
            for cust in range(num_custs):
                added = False
                while not added:
                    x,y = _locate_customer(self.rng_inst, size, customer_dist, big=big, small=small)
                    if (x, y) not in customers and (x, y) != depot_pos and 0 <= x < size and 0 <= y < size:
                        customers.append((x, y))
                        added = True
                customer_times.append(i)
        
        # if a customer's location makes it such that there is no way it could be served in time, then make it so that it never requests
        # (essentially, just ignore impossible customers)
        for pos,(i,time) in zip(customers, enumerate(customer_times)):
            if time + self._manhattan_distance(depot_pos, pos) > remaining_time:
                customer_times[i] = remaining_time
        
        # NOTE: customers and depot are distributed amongst the size x size grid, but agent can always explore the max size (40x40), even if smaller size (30x30) chosen
        
        self.game = Game(self.type, self.rwtype, (size, size), depot_pos, depot_pos, remaining_time, customers, customer_times, self.mask)

    def render(self,render_mode=None):
        return self.game.render(mode=render_mode)

    def close(self):
        pass

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) # TODO could be improved: np.linalg.norm(p1-p2, 1)

    def get_max_score(self):
        return self.game.max_score

    def get_total_custs(self):
        return self.game.total_servable_custs
    
    def get_episode_summary(self):
        summary = {}
        summary['reqs_served'] = len([c for c in self.game.client_list.values() if c.served])
        summary['total_reqs'] = len([t for t in self.game.client_times.values() if t < self.game.total_time])
        # wait times include 0s for customers that never requested; this is fine since
        # we're reporting total wait times
        wait_times = [
            ((c.time_served if c.time_served is not None else self.game.total_time) - self.game.client_times[c.pos])
            for c in self.game.client_list.values()]
        summary['total_wait_time'] = sum(wait_times)
        return summary
