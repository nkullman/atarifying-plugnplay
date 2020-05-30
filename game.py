import numpy as np
import npdraw
import rewards


PROB_RATIO = 0.1
REWARD_PROP_RATE = 0.90
REWARD_PROPAGATION_PROPORTION = 0.1


class Client:
    def __init__(self, pos):
        self.pos = pos
        self.requested = False
        self.served = False
        self.hide = False
        self.time_served = None


RPS = {
    "custom" : rewards.MyRPS,
    "paper" : rewards.PaperRPS,
    "paperlike" : rewards.PaperlikeRPS,
    "linear" : rewards.LinearPenaltyRPS,
    "piecewise" : rewards.PiecewisePenaltyRPS,
    "quadratic" : rewards.QuadraticPenaltyRPS,
    "reward": rewards.RewardBasedRPS,
    "mask": rewards.ActionMaskRPS,
    "ndk": rewards.NdkRPSBase,
    "ndkp": rewards.NdkRPSWaitPenalty,
    "ndkn": rewards.NdkRPSPenaltiesOnly
}

MOVES = ((-1,0), (0,1), (1,0), (0,-1), (0, 0)) # in array space: up, right, down, left, NOOP (in coord space: left, up, right, down, NOOP)
OPEN_MASK = [True, True, True, True, True]

class Game:

    def __init__(self, rtype, rwtype, shape, depot_pos, car_pos, remaining_time, client_list=None, client_times=None, mask=False):
        
        assert len(shape) == 2 and len(depot_pos) == 2
        self.shape = shape
        if not (0 <= depot_pos[0] < shape[0] and 0 <= depot_pos[1] < shape[1]):
            raise IndexError("Depot must be in the grid")
        self.depot_pos = tuple(depot_pos)
        self.car_pos = tuple(car_pos)
        self.client_times = {}
        self.client_list = {}
        for k, c in enumerate(list(client_list)) if client_list is not None else []:
            cc = tuple(c)
            if not cc in self.client_list and cc != depot_pos:
                cl = Client(cc)
                if not client_times:
                    if cc != car_pos and np.random.rand() < PROB_RATIO * 2:
                        cl.requested = True
                else:
                    cl.requested = True if client_times[k] == 0 else False
                    self.client_times[cc] = client_times[k]
                self.client_list[cc] = cl
        self.remaining_time = remaining_time
        self.total_time = remaining_time
        self.servable_custs = [c_pos for c_pos,t in self.client_times.items() if t < self.total_time]
        self.total_servable_custs = len(self.servable_custs)
        self.last_render = None
        self.last_render_p = None
        self.last_action = None
        self.it_after_end = 0
        self.type = rtype
        self.rps = RPS[rwtype](self)
        self.score = 0
        self.mask = mask
        self.depot_dists = self._get_dists_from_depot()
        self.max_score = self.rps.max_score
        self.min_score = self.rps.min_score

    def _get_dists_from_depot(self):
        
        dists = np.zeros(self.shape)
        xdists = np.abs(np.arange(self.shape[0]) - self.depot_pos[0])[None,:]
        ydists = np.abs(np.arange(self.shape[1]) - self.depot_pos[1])[:,None]
        dists = (dists + xdists) + ydists
        return dists
    
    def _get_mask(self):
        if not self.mask:
            return OPEN_MASK
        else:
            return [
            self.remaining_time > self._manhattan_distance(self.depot_pos, (self.car_pos[0] - 1, self.car_pos[1])),
            self.remaining_time > self._manhattan_distance(self.depot_pos, (self.car_pos[0], self.car_pos[1] + 1)),
            self.remaining_time > self._manhattan_distance(self.depot_pos, (self.car_pos[0] + 1, self.car_pos[1])),
            self.remaining_time > self._manhattan_distance(self.depot_pos, (self.car_pos[0], self.car_pos[1] - 1)),
            self.remaining_time > self._manhattan_distance(self.depot_pos, self.car_pos)
        ]

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) # TODO could be improved: np.linalg.norm(p2-p1, 1)

    def _get_disappearing_customers(self):
        # those disappearing are those that are not already served or hidden and 
        # cannot be feasibly reached by the vehicle in the remaining time
        return [
            c for c in self.client_list.values() if (
                not (c.served or c.hide) and 
                self._manhattan_distance(self.car_pos,c.pos) + self._manhattan_distance(c.pos,self.depot_pos) > self.remaining_time)]
    
    @property
    def curr_time(self):
        return self.total_time - self.remaining_time
    
    def step(self, action):
        
        # Move the car and ensure that it is on the field
        assert 0 <= action < 5, f"Invalid action value: {action}"
        move = MOVES[action]
        
        # get the reward, specify if the next state is terminal
        reward, terminal = self.rps.reward(move)

        self.remaining_time -= 1

        # Update requesting clients
        for k, c in self.client_list.items():
            # for the game instances that don't come with pre-defined client release times
            if not self.client_times:
                if not c.requested and c.pos != self.car_pos:
                    if np.random.rand() < PROB_RATIO:
                        c.requested = True
            
            # for the paper-based instances that have client times pre-defined
            else:
                # mark customers that are currently requesting
                if self.client_times[k] <= self.curr_time:
                    c.requested = True

        self.car_pos = (
            self.car_pos[0] + move[0] if 0 <= self.car_pos[0] + move[0] < self.shape[0] else self.car_pos[0],
            self.car_pos[1] + move[1] if 0 <= self.car_pos[1] + move[1] < self.shape[1] else self.car_pos[1])
        
        # Serve
        if self.car_pos in self.client_list:
            c = self.client_list[self.car_pos]
            if c.requested and not (c.served or c.hide):
                c.served = True
                c.time_served = self.curr_time

        # if getting close to the end of the horizon, mark those customers that can no longer be served in time
        # only need to do this check when time is getting close.
        if 0 <= self.remaining_time <= self.shape[0]*2:
            disappearing_custs = self._get_disappearing_customers()
            for c in disappearing_custs:
                c.hide = True

        if self.remaining_time < 0:
            self.it_after_end += 1

        if self.it_after_end > (self.shape[0] * self.shape[1]):
            terminal = True # force termination if it's really taking forever

        self.score += reward
        mask = self._get_mask()
        return self.render(), reward, terminal, mask # state, reward, terminal, info (mask)


    def render(self, mode=None):
        """Renders the current state of the game.
        Valid modes are 'binarystack', 'pixelgray', 'pixelrgb','pixelmask'."""
        
        render_mode = self.type if mode is None else mode

        if render_mode == "drawn":
            return self.render_drawn()
        
        elif render_mode == "pixelgray":
            return self.render_pixel(mode='grayscale')
        
        elif render_mode == "pixelrgb":
            return self.render_pixel(mode="rgb")
        
        elif render_mode == "pixelmask":
            return self.render_pixel(mode="grayscale", mask=True)
        
        elif render_mode == "binarystack":
            return self._pixel_render_binary_stack()
        
        elif render_mode == 'classic':
            return self.render_classic()
        
        else:
            raise ValueError("Invalid render type")

    # def render_drawn(self):
    #     raise NotImplementedError("Drawn render not currently supported.")
    #     if self.last_render is not None:
    #         return self.last_render
    #     canvas = np.zeros([84, 84], dtype=np.uint8)
    #     npdraw.clear(canvas)
    #     # Background
    #     for i in range(self.shape[0]):
    #         npdraw.horizontal_line(canvas, 6, 77, self._node_pos_drawn((i, 0))[0], color=192)
    #     for i in range(self.shape[1]):
    #         npdraw.vertical_line(canvas, self._node_pos_drawn((0, i))[1], 6, 77, color=192)
    #     # Entities
    #     npdraw.square(canvas, self._node_pos_drawn(self.depot_pos), 3, color=128, border_color=64)
    #     for _, c in self.client_list.items():
    #         if not c.requested:
    #             npdraw.lozenge(canvas, self._node_pos_drawn(c.pos), 3, color=128, border_color=64)
    #         elif c.requested and not c.served:
    #             npdraw.lozenge(canvas, self._node_pos_drawn(c.pos), 3, color=0, border_color=64)
    #         else:
    #             npdraw.lozenge(canvas, self._node_pos_drawn(c.pos), 3, color=255, border_color=64)
    #     npdraw.square(canvas, self._node_pos_drawn(self.car_pos), 1, color=255, border_color=0)
    #     # Timer
    #     timer_width = 84 * self.remaining_time // self.total_time
    #     npdraw.horizontal_line(canvas, 0, timer_width-1, 0)
    #     npdraw.horizontal_line(canvas, 0, timer_width-1, 1)
    #     return canvas

    # def _node_pos_drawn(self, pos):
    #     assert len(pos) == 2
    #     return (6 + ((71 * pos[0]) // (self.shape[0] - 1)),
    #             6 + ((71 * pos[1]) // (self.shape[1] - 1)))

    def render_pixel(self, mode='grayscale', mask=False):
        """Provides a pixel-based representation of the current game state."""
        
        if mode=='rgb':
            return self._pixel_render_colored(mask,grayscale=False)
        
        else:
            return self._pixel_render_colored(mask)
        
    # def _node_pos_pixel(self, pos):
    #     return (pos[0] + 16 - self.shape[0] // 2,
    #             pos[1] + 16 - self.shape[1] // 2)
    
    def _pixel_render_colored(self, mask, grayscale=True):
        """Print a human-viewable version of the state of the game.
        Default is grayscale. Set to False for RGB.
        """

        # define colors of objects in the frame
        if grayscale:
            # numerically distinct "colors" in the grayscale spectrum
            colors = {
                'base': 0, # black base
                'depot':80,
                'cust_potential':100,
                'car':255, # white car
                'cust_active':240
            }
        
        else:
            # visually distinct colors in the RGB spectrum
            colors = {
                'base': 224, # light gray base
                'depot':(51, 51, 255), # depot:  blue
                'cust_potential':(255, 187, 51), # potential customers: yellow
                'car':(255, 51, 153), # car: pink
                'cust_active':(119, 255, 51) # active customers: green
            }
        
        # drawing order:
        # base, depot, potential, car, active
        
        # establish canvas, color the base
        canvas = np.ones(
            shape=(self.shape if grayscale else (self.shape+(3,))),
            dtype=np.uint8) * colors['base']
        
        # color the depot
        canvas[self.depot_pos[0],self.depot_pos[1],...] = colors['depot']
        
        # potential customers
        for c in self.client_list.values():
            if not c.hide:
                if not c.served and not c.requested: # unrequested and unserved
                    canvas[c.pos[0],c.pos[1],...] = colors['cust_potential']
        
        # car is a 3x3 marking (with its middle hollowed out)
        # when the car is on the edge, its mark will spill out into the buffer,
        # so we'll add that before coloring the car
        buff_width = 1 if self.shape[0] <= 20 else 2
        buff_color = rbg_to_gray((40,40,40)) # a dark gray border
        if grayscale:
            canvas = np.pad(canvas, pad_width=((buff_width,buff_width),), mode='constant', constant_values=buff_color)
        else:
            canvas = np.pad(canvas, pad_width=((buff_width,buff_width),(buff_width,buff_width),(0,0)), mode='constant', constant_values=buff_color)

        # canvas added, color the car's mark
        self._add_car_to_canvas(self.car_pos, colors['car'], canvas, buff_width)

        # active customers
        for c in self.client_list.values():
            if not c.hide:
                if not c.served and c.requested: # requested and unserved
                    canvas[c.pos[0]+buff_width,c.pos[1]+buff_width,...] = colors['cust_active']
        
        canvas = np.rot90(canvas,k=1,axes=(0,1))
        canvas = self._add_time_bar(canvas, buff_color, 100)

        return canvas
    
    def _add_car_to_canvas(self,car_pos,car_color,canvas,w_buff):
        center = [car_pos[0]+w_buff, car_pos[1]+w_buff]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i==j==0:
                    continue # leave the middle open
                canvas[center[0]+i, center[1]+j,...] = car_color
        return canvas
    
    def _add_time_bar(self,frame,base_color,bar_color):
        """Returns an augmented frame where a time bar has been added to its "top" (visually).
        There is a row depicting the amount of time left (white/1 for the pixels indicating expired time, black/0 for still remaining).
        """
        time_bar = (np.arange(frame.shape[0]) < ((self.remaining_time/self.total_time) * frame.shape[0])).astype(np.uint8)
        time_bar = np.where(time_bar>0,bar_color,base_color)[None,:]
        
        # thicker time bar for larger renderings (NOTE should be set more dynamically)
        if self.shape[0] >= 20:
            time_bar = np.tile(time_bar,(2,1))

        if frame.ndim == 2:
            # grayscale
            return np.vstack([time_bar,frame])
        else:
            # rgb
            return np.concatenate([np.tile(time_bar[...,None],(1,1,3)),frame],axis=0)

    
    def _pixel_render_binary_stack(self):
        
        canvas = np.zeros([3, self.shape[0], self.shape[1]], dtype=np.float32) # arrays: car, client.potential, client.active; scalar: time
        # Entities
        car = self.car_pos
        depot = self.depot_pos
        canvas[0,car[0],car[1]] = 1
        for c in self.client_list.values():
            # ignore served customers
            if not (c.served or c.hide):
                pos = c.pos
                # active customer
                if c.requested:
                    canvas[2,pos[0],pos[1]] = 1
                # potential customer not yet requesting
                else:
                    canvas[1,pos[0],pos[1]] = 1
        return canvas, self.remaining_time / self.total_time

    def render_classic(self):
        return {
            'car':self.car_pos,
            'depot':self.depot_pos,
            'curr_cust':[c.pos for c in self.client_list.values() if (c.requested and not (c.served or c.hide))],
            'potential_cust':[c.pos for c in self.client_list.values() if (not c.requested and not (c.served or c.hide))],
            'time':self.curr_time,
            'remaining_time':self.remaining_time
        }
    
    def get_pi_state(self):
        return {
            'car':self.car_pos,
            'depot':self.depot_pos,
            'curr_cust':[c.pos for c in self.client_list.values() if (c.requested and not (c.served or c.hide))],
            'potential_cust':[c.pos for c in self.client_list.values() if (not c.requested and not (c.served or c.hide))],
            'time':self.curr_time,
            'remaining_time':self.remaining_time,
            'pi_requests': [
                {
                    's':c_pos,
                    't':c_pos,
                    'time':self.client_times[c_pos],
                    'time_to_depot':self.depot_dists[c_pos]
                } for c_pos in self.servable_custs ]
        }
    
def rbg_to_gray(rgb):
    assert len(rgb)==3, "Invalid rgb array passed"
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]


if __name__ == "__main__":
    game = Game('drawn', (5,6), (2,2), (3,3), 20, [(1,1), (4,3), (3,1), (1,3)])
    #imageio.imwrite("img_out/out.png", game.render())
