import numpy as np

from vrpssratari import utils

class Customer:
    """Defines a customer in the Atari-fied VRPSSR game.
        
        Fields:
            pos (tuple): the (x,y) coords of the customer
            requested (bool): whether the customer has begun requesting service
            served (bool): whether the customer has been served
            hide (bool): whether the customer is currently hidden
            time_served (int or None): time at which the customer was served
    
    """
    def __init__(self, pos):
        self.pos = pos
        self.requested = False
        self.served = False
        self.hide = False
        self.time_served = None

class VrpssrGame:
    """Defines the Atari-fied VRPSSR game."""

    _SERVICE_REWARD = 10
    
    def __init__(self, game_config, rng):
        self.shape = game_config['shape']
        self.depot_pos = game_config['depot']
        self.init_car_pos = game_config['car']
        self.game_length = game_config['game_length']
        self.cust_dist_type = game_config['cust_dist_type']
        self.exp_n_cust = game_config['exp_n_cust']
        self.exp_n_cust0 = game_config['exp_n_cust0']
        self.exp_n_cust_later = self.exp_n_cust - self.exp_n_cust0
        self._clusters = {
            'means':{
                'cluster2': [
                    (self.shape[0]/4, self.shape[1]/4),
                    (self.shape[0]/4, 3*self.shape[1]/4)],
                'cluster3': [
                    (self.shape[0]/4, self.shape[1]/4),
                    (self.shape[0]/4, 3*self.shape[1]/4),
                    (3*self.shape[0]/4, self.shape[1]/2)]
            },
            'wts':{
                'cluster2': [0.5,0.5],
                'cluster3': [0.25,0.5,0.25]
            }
        }
        self.rng = rng

        self.car_pos = None
        self.cust_req_times = {} # dict: pos tuple to time of req
        self.custs = {} # dict: pos to customer object
        self.remaining_time = -1
        self.servable_custs = None
        self.num_servable_custs = -1
        self.score = float('nan')
        self.done = True

    def _locate_customer(self):
        """Places a customer on the game board following the method described
        in Ulmer et al (2017).
        """
        
        if self.cust_dist_type == 'random':
            return (self.rng.randint(shape[0]), self.rng.randint(shape[1]))
        
        else:
            
            cluster_means = self._clusters['means'][self.cust_dist_type]
            num_clusters = len(cluster_means)
            wts = self._clusters['wts'][self.cust_dist_type]
            # draw a cluster in which to place the customer
            means = cluster_means[self.rng.choice(range(num_clusters), p=wts)]

            # randomly draw x,y near the cluster
            # (not exactly like Ulmer et al, since this will have stddev sqrt(2), but close enough. 
            # sort of necessary anyways, since we're fishing for unique int pairs, which isn't as easy as unique float pairs)
            return int(np.rint(self.rng.normal(means[0],2))), int(np.rint(self.rng.normal(means[1],2)))
    
    def generate_game(self):
        
        # initizlize the number of customers newly requesting service for each time interval
        num_requesting = np.zeros((self.game_length,), dtype=np.int32)
        
        num_requesting[0] = self.rng.poisson(self.exp_n_cust0) # number of customers that will be requesting service initially (at t=0)
        num_requesting[1:] = self.rng.poisson(self.exp_n_cust_later/(self.game_length-1), size=(self.game_length-1))
        
        for t, num_custs in enumerate(num_requesting):
            for cust in range(num_custs):
                added = False
                while not added:
                    x,y = self._locate_customer()
                    if (x, y) not in self.custs and (x, y) != self.depot_pos and 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
                        cl = Customer((x,y))
                        cl.requested = t == 0 # if requesting at t=0, then set to true; otherwise false
                        self.custs[(x,y)] = cl
                        self.cust_req_times[(x,y)] = t
                        added = True
        
        # if a customer's location makes it such that there is no way it could be served in time, then make it so that it never requests
        # (essentially, just ignore impossible customers)
        for cpos in self.cust_req_times.keys():
            t = self.cust_req_times[cpos]
            if t + utils.dist(cpos,self.depot_pos) > self.game_length:
                self.cust_req_times[cpos] = self.game_length
        
        self.car_pos = self.init_car_pos
        self.remaining_time = self.game_length
        self.servable_custs = [c_pos for c_pos,t in self.cust_req_times.items() if t < self.game_length]
        self.num_servable_custs = len(self.servable_custs)
        self.score = 0
        self.done = False

    def _get_disappearing_customers(self):
        """Disappearing custs are those that are not already served or hidden and 
        cannot be feasibly reached by the vehicle in the remaining time"""

        return [
            c for c in self.custs.values() if (
                not (c.served or c.hide) and 
                utils.dist(self.car_pos,c.pos) + utils.dist(c.pos,self.depot_pos) > self.remaining_time)]

    @property
    def curr_time(self):
        return self.game_length - self.remaining_time

    def advance(self, move):
        
        reward = 0                  # initialize the reward
        self.remaining_time -= 1    # note the new time
        
        # note the car's new position
        self.car_pos = (
            self.car_pos[0] + move[0] if 0 <= self.car_pos[0] + move[0] < self.shape[0] else self.car_pos[0],
            self.car_pos[1] + move[1] if 0 <= self.car_pos[1] + move[1] < self.shape[1] else self.car_pos[1])

        # Update requesting customers 
        for k, c in self.custs.items():
            # mark customers that are currently requesting
            if self.cust_req_times[k] <= self.curr_time:
                c.requested = True
        
        # Note if our vehicle has served a new customer
        if self.car_pos in self.custs:
            c = self.custs[self.car_pos]
            if c.requested and not (c.served or c.hide):
                c.served = True
                c.time_served = self.curr_time
                reward += self._SERVICE_REWARD

        # if getting close to the end of the horizon, mark customers that can no longer be served in time
        # only need to do this check when time is getting close.
        if 0 <= self.remaining_time <= self.shape[0]+self.shape[1]:
            disappearing_custs = self._get_disappearing_customers()
            for c in disappearing_custs:
                c.hide = True

        # note whether game is over (ends when vehicle has no more time to
        # explore bc it has to return to the depot)
        self.done = utils.dist(self.car_pos, self.depot_pos) >= self.remaining_time
        
        self.score += reward

        return reward
