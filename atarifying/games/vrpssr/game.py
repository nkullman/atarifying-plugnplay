import numpy as np

from atarifying import utils

class VrpssrGame:
    """Defines the Atari-fied VRPSSR game."""

    _SERVICE_REWARD = 10
    
    def __init__(self, game_config, rng):
        self.shape = game_config['shape']
        self._upper_extents = (self.shape[0]-1, self.shape[1]-1) # max vals that indices can take
        self.depot_pos = np.array(game_config['depot'])
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
        self.cust_req_times = None
        self.custs = None
        self.remaining_time = -1
        self.score = float('nan')
        self.done = True
        self._dists_to_depot = self._get_dists_from_depot()

    def _get_dists_from_depot(self):
        dists = np.zeros(self.shape, dtype=np.int)
        xdists = np.abs(np.arange(self.shape[0]) - self.depot_pos[0])[None,:]
        ydists = np.abs(np.arange(self.shape[1]) - self.depot_pos[1])[:,None]
        dists = (dists + xdists) + ydists
        return dists

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
        
        # initizlize the number of customers newly requesting service for each time period
        num_requesting = np.zeros((self.game_length,), dtype=np.int32)
        
        num_requesting[0] = self.rng.poisson(self.exp_n_cust0) # number of customers that will be requesting service initially (at t=0)
        num_requesting[1:] = self.rng.poisson(self.exp_n_cust_later/(self.game_length-1), size=(self.game_length-1))

        self.custs = np.zeros(shape=self.shape, dtype=np.bool)                          # locations with customers
        self.req_times = np.ones(shape=self.shape, dtype=np.float) * self.game_length   # request time for each location
        
        for t, num_custs in enumerate(num_requesting):
            for cust in range(num_custs):
                added = False
                while not added:
                    loc = self._locate_customer() # loc is (x,y) tuple
                    # if proposed customer location is a new one on the board (and not on top of depot)...
                    if ~self.custs[loc] and np.any(self.depot_pos != loc) and 0 <= loc[0] < self.shape[0] and 0 <= loc[1] < self.shape[1]:
                        self.custs[loc] = True          # then mark its spot on the custs array
                        self.req_times[loc] = t         # mark its requesting time
                        added = True                    # and mark it as added
        
        # all custs now added
        self.requested = self.req_times == 0                                            # mark which ones are requesting now
        self.served = np.zeros(shape=self.shape, dtype=np.bool)                         # mark that none of them have been served
        self.serve_times = np.ones(shape=self.shape, dtype=np.float) * self.game_length # and say that all service times are at the deadline )
        
        self.car_pos = np.array(self.init_car_pos)
        self.remaining_time = self.game_length
        self.score = 0
        self.done = False

    @property
    def curr_time(self):
        return self.game_length - self.remaining_time

    def advance(self, move):
        
        self.remaining_time -= 1    # note the new time
        
        # note the car's new position
        self.car_pos = np.clip(self.car_pos + move, (0,0), self._upper_extents)
        car_pos_tup = tuple(self.car_pos)

        # Update requesting customers 
        self.requested = self.req_times <= self.curr_time
        
        # Note if our vehicle has served a new customer
        requesting = self.requested & ~self.served                  # get which locations are active custs
        reward = self._SERVICE_REWARD * requesting[car_pos_tup]     # if we're at one, earn a positive reward; otherwise 0
        if requesting[car_pos_tup]:
            self.served[car_pos_tup] = True                         # if we're serving, mark cust as served
            self.serve_times[car_pos_tup] = self.curr_time          # and note the time we served it

        # game ends when it has to head straight back to depot
        self.done = self._dists_to_depot[car_pos_tup] > self.remaining_time
        
        self.score += reward

        return reward
