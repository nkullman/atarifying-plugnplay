from collections import deque
import logging

import gym
import numpy as np

import utils
import game

gym.envs.registration.register(
    id='Vrpssr-v0',
    entry_point='environment:VrpssrEnv'
)

class VrpssrEnv(gym.Env):
    """Defines an environment for playing the Atari-fied VRPSSR game."""
    
    metadata = {
        'render.modes': ['human']
    }

    _MOVES = [(-1,0), (0,1), (1,0), (0,-1), (0, 0)]
    _ONE_FRAME_STATE_TYPES = ['feature_layers', 'feature_layers_nonnorm', 'classic', 'human']
    _BOARD_BUFFER_WIDTH = 1
    _BOARD_TIMEBAR_HEIGHT = 2

    def __init__(self, env_config):
        """Initialize an Environment that is compatible with a game representing
        the Atarified VRPSSR.

        Args:
            env_config (dict): A configuration for the environment (see ex below)

        Example env_config:
        {
            'state_type': 'feature_layers',      # what type of state the agent should receive
            'n_frames': 1,                      # how many previous videogame 'frames' to include in the state (NOTE these are not displayed in state)
            'seed': null,                       # seed for game randomness
            'shape': (32,32),                   # the size of the game board
            'depot': (16,16),                   # the location of the depot
            'car': (16,16),                     # the location where the vehicle begins
            'game_length': 230,                 # duration of a game
            'cust_dist_type':'cluster3',        # how customers are distributed on the game board
            'exp_n_cust': 30,                   # avg num customers in a game
            'exp_n_cust0': 15,                  # avg num customers requesting service at t=0
        }
        # TODO add param for how much time one step consumes
        """

        self.state_type = env_config.get('state_type', 'feature_layers')
        self.seed = env_config.get('seed', None)
        self.rng = np.random.RandomState(seed=self.seed)
        self.n_frames = env_config.get('n_frames', 1)
        if self.state_type in self._ONE_FRAME_STATE_TYPES:
            if self.n_frames > 1:
                logging.warning(f"State type {self.state_type} only supports single-frame state.")
            self.n_frames = 1 # for feature_layers and classic, the concept of "frames" does not apply
        self.frame_stack = deque(maxlen=self.n_frames) # holds the last nframes number of game renders
        # NOTE we refer to game renderings as frames (bc they are often images), but they may not necessarily be
        # for example, in the 'classic' render mode, a game frame is a dictionary of relevant data for an agent
        # to make a decision: vehicle location, current time, depot location, locations of current (active) requests,
        # and locations of potential requests
        
        
        self.game_config = {
            'shape': env_config.get('shape', (32,32)),
            'depot': env_config.get('depot', (16,16)),
            'car': env_config.get('car', (16,16)),
            'game_length': env_config.get('game_length', 230), # set in accordance with the len/time ratio of 360min:20km = 360 min:40 grid spaces
            'cust_dist_type': env_config.get('cust_dist_type', 'cluster3'),
            'exp_n_cust': env_config.get('exp_n_cust', 30),
            'exp_n_cust0': env_config.get('exp_n_cust0', 15),
        }
        
        # action space is the set of all possible moves
        self.action_space = gym.spaces.Discrete(len(self._MOVES))
        
        # state space is the set of all possibly observable states
        self.observation_space = self._get_observation_space()

        self._game = None

        self.curr_state = None
        self._last_game_summary = None

    def reset(self):
        """Launches a new game.

        Returns:
            obs: The inital state of the environment
        """
        
        # initialize a game
        self._game = game.VrpssrGame(self.game_config,self.rng)
        self._game.generate_game()
        
        # reset the stack of most recent game frames
        self.frame_stack.clear()

        # retrieve the first game frame, append it to the frame stack
        new_frame = self._show_game(mode=self.state_type)
        self.frame_stack.append(new_frame)
        
        # get the state from the stack of game frames
        self.curr_state = self._get_state_from_frames()
        
        return self.curr_state

    def step(self, action):
        """Advances the environment based on the new action

        Args:
            action (int): The action taken from the current game state

        Returns:
            tuple: next_state, reward, is_terminal, info
        """
        assert self.curr_state is not None, "Game must be reset in order to step"

        move = self._MOVES[action] # game's advance function expects the move, not the action index
        reward = self._game.advance(move)

        new_frame = self._show_game(mode=self.state_type)
        self.frame_stack.append(new_frame)

        self.curr_state = self._get_state_from_frames()

        if self._game.done:
            wait_times = [
                ((c.time_served if c.time_served is not None else self._game.game_length) - self._game.cust_req_times[c.pos])
                for c in self._game.custs.values()]

            self._last_game_summary = {
                'reqs_served': len([c for c in self.game.client_list.values() if c.served]),
                'total_reqs': len([t for t in self.game.client_times.values() if t < self.game.total_time]),
                'total_wait_time': sum(wait_times)
            }

        return self.curr_state, reward, self._game.done, {}

    def render(self, mode='human'):
        """Provides a depiction of the current game state

        Args:
            mode (str, optional): Type of depiction desired. Defaults to 'human'.

        Returns:
            any: The depiction of the environment
        """
        return self._show_game(mode=mode)

    def close(self) -> None:
        pass

    def get_episode_summary(self) -> dict:
        """Retrieve stats about the last completed game.

        Raises:
            ValueError: If no game has been completed.

        Returns:
            dict: Summary stats of the last completed game. Keys are metric names, values are their values.
        """
        
        if self._last_game_summary is None:
            raise ValueError("No completed game information available.")
        return self._last_game_summary

    def _get_state_from_frames(self):
        """Given the current memory of game states in the frame stack, provides an observation
        compatible with the observation_space.

        Returns:
            any: An observation of the current game state
        """
        
        if self.state_type in self._ONE_FRAME_STATE_TYPES:
            return self.frame_stack[-1] # return just the most recent "frame"
        
        else:
            # returning a stack of frames
            state = np.array(self.frame_stack)
            # if we don't have as many frames as would be expected, then we repeat the most recent frame
            if state.shape[0] < self.n_frames:
                rep_times = self.n_frames - len(self.frame_stack) # how many frames are we missing?
                state = np.pad(state, pad_width=[(0,rep_times),(0,0),(0,0)], mode='edge') # repeat the last one that many times
            return state

    def _get_observation_space(self):
        """Defines the observation space for the configuration.

        Raises:
            ValueError: If the environment was configured with an invalid state type

        Returns:
            gym.Space: The observation space
        """

        # for drawn state types (human and humangray), the shape needs to include the border and time bar
        drawn_game_size = np.array(self.game_config['shape']) + 2*self._BOARD_BUFFER_WIDTH      # add borders
        drawn_game_size[1] = drawn_game_size[1] + self._BOARD_TIMEBAR_HEIGHT                    # and time bar
        drawn_game_shape = tuple(drawn_game_size)

        if self.state_type == 'human':
            # game board, with a layer for RGB
            return gym.spaces.Box(low=0, high=255, shape=(drawn_game_shape + (3,)), dtype=np.int32)
        
        elif self.state_type == 'humangray':
            # the last n_frames game boards
            return gym.spaces.Box(low=0, high=255, shape=((self.n_frames,) + drawn_game_shape), dtype=np.int32)
        
        elif self.state_type == 'feature_layers':
            # (game board for [vehicle, potential & active custs], relative time remaining)
            return gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=1, shape=((3,) + self.game_config['shape']), dtype=np.int32),
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            ))
        
        elif self.state_type == 'feature_layers_nonnorm':
            # same as feature_layers, except the values in the feature layers are 0,255 rather than 0,1
            return gym.spaces.Tuple((
                gym.spaces.Box(low=0, high=255, shape=((3,) + self.game_config['shape']), dtype=np.int32),
                gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            ))
        
        elif self.state_type == 'classic':
            # define the max values positions can take
            space_extent = np.array([self.game_config['shape'][0],self.game_config['shape'][1]]) - 1
            return gym.spaces.Dict({
                'car':gym.spaces.Box(low=np.array([0,0]), high=space_extent, dtype=np.int32),
                'depot':gym.spaces.Box(low=np.array([0,0]), high=space_extent, dtype=np.int32),
                'curr_cust':gym.spaces.Box(low=0, high=1, shape=((3,) + self.game_config['shape']), dtype=np.int32),
                'potential_cust':gym.spaces.Box(low=0, high=1, shape=((3,) + self.game_config['shape']), dtype=np.int32),
                'time':gym.spaces.Box(low=0, high=self.game_config['game_length'], shape=(1,), dtype=np.int32),
                'remaining_time':gym.spaces.Box(low=0, high=self.game_config['game_length'], shape=(1,), dtype=np.int32)
            })
        
        else:
            raise ValueError(f"Unsupported state type: {self.state_type}")

    def _show_game(self, mode):
        """Provide a depiction of the current game state.

        Args:
            mode (str): Type of state depiction to use. Options: 'humangray', 'human', 'feature_layers', 'feature_layers_nonnorm', and 'classic'

        Raises:
            ValueError: If an invalid mode type is provided

        Returns:
            any: The mode-dependent state depiction
        """

        if mode == "humangray":
            return self._render_pixel(mode='grayscale')
        
        elif mode == "human":
            return self._render_pixel(mode="rgb")
        
        elif mode == "feature_layers":
            return self._render_feature_layers()
        
        elif mode == "feature_layers_nonnorm":
            return self._render_feature_layers(normalize=False)
        
        elif mode == 'classic':
            return self._render_classic()
        
        else:
            raise ValueError("Invalid render type")
    
    def _render_pixel(self, mode='grayscale'):
        """Provides a visual (pixel-based) representation of the current game state.

        Args:
            mode (str, optional): Type of coloring to use ('rgb' or 'grayscale'). Defaults to 'grayscale'.

        Returns:
            np.array: A canvas depicting the current game state. Canvas is 3D if mode=='rgb' (2D otherwise)
        """
        
        grayscale = mode != 'rgb'
        
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
            shape=(self._game.shape if grayscale else (self._game.shape+(3,))),
            dtype=np.uint8) * colors['base']
        
        # color the depot
        canvas[self._game.depot_pos[0],self._game.depot_pos[1],...] = colors['depot']
        
        # potential customers
        for c in self._game.custs.values():
            if not (c.served or c.hide) and not c.requested: # unrequested and unserved
                canvas[c.pos[0],c.pos[1],...] = colors['cust_potential']
        
        # car is a 3x3 marking (with its middle hollowed out)
        # when the car is on the edge, its mark will spill out into the buffer,
        # so we'll add that before coloring the car
        buff_width = self._BOARD_BUFFER_WIDTH
        buff_color = 0 # let the border be black
        if grayscale:
            canvas = np.pad(canvas, pad_width=((buff_width,buff_width),), mode='constant', constant_values=buff_color)
        else:
            canvas = np.pad(canvas, pad_width=((buff_width,buff_width),(buff_width,buff_width),(0,0)), mode='constant', constant_values=buff_color)

        # canvas added, color the car's mark
        self._add_car_to_canvas(self._game.car_pos, colors['car'], canvas, buff_width)

        # active customers
        for c in self._game.custs.values():
            if not c.hide:
                if not c.served and c.requested: # requested and unserved
                    canvas[c.pos[0]+buff_width,c.pos[1]+buff_width,...] = colors['cust_active']
        
        canvas = np.rot90(canvas,k=1,axes=(0,1))
        
        time_bar_color = 100
        canvas = self._add_time_bar(canvas, buff_color, time_bar_color)

        return canvas
    
    def _add_car_to_canvas(self, car_pos, car_color, canvas, w_buff):
        """Adds the car to the canvas. The car is represented as a 3x3 square with a hole in the middle.

        Args:
            car_pos (tuple): (x,y) of the car's location
            car_color (int or tuple): The color of the car. Either an (r,g,b) tuple or an int (grayscale or identical rgb vals)
            canvas (np.array): The canvas on which to add the car
            w_buff (int): The width of the buffer surrounding the playable area on the canvas

        Returns:
            np.array: The canvas with the vehicle.
        """
        
        # where is the car centered
        center = [car_pos[0]+w_buff, car_pos[1]+w_buff]
        # retrieve the canvas's current color there
        center_color = canvas[tuple(center)]
        # color all the surrounding elements (incl the center)
        canvas[center[0]-1:center[0]+2, center[1]-1:center[1]+2,...] = car_color
        # "hollow out" the center by restoring it to its previous value
        canvas[tuple(center)] = center_color
        
        return canvas
    
    def _add_time_bar(self, canvas, base_color, bar_color):
        """Returns a canvas where a time bar has been added to its "top" (visually).

        Args:
            canvas (np.array): The canvas on which to draw the time bar
            base_color (int): Color value of the part of the time bar that is depleted
            bar_color (int): Color value of the part of the time bar that remains

        Returns:
            np.array: A canvas with an added time bar
        """
        
        time_bar = (np.arange(canvas.shape[0]) < ((self._game.remaining_time/self._game.game_length) * canvas.shape[0])).astype(np.uint8)
        time_bar = np.where(time_bar>0, bar_color, base_color)[None,:] # establish a single-row version of the time bar
        time_bar = np.tile(time_bar, (self._BOARD_TIMEBAR_HEIGHT,1)) # make it the desired height
        
        if canvas.ndim == 2:
            # grayscale
            return np.vstack([time_bar,canvas])

        else:
            # rgb
            return np.concatenate([np.tile(time_bar[...,None],(1,1,3)),canvas],axis=0)

    def _render_feature_layers(self,normalize=True):
        """Provides a feature-layer representation of the current game state

        Args:
            normalize (bool, optional): Whether all output values should be in [0,1] (otherwise in [0,255]). Defaults to True.

        Returns:
            tuple (np.array, float): [feature-layers for the vehicle, potential customers, active customers], the relative time remaining
        """
        
        canvas = np.zeros([3, self._game.shape[0], self._game.shape[1]], dtype=np.float32) # arrays: car, client.potential, client.active; scalar: time
        # Entities
        car = self._game.car_pos
        depot = self._game.depot_pos
        canvas[0,car[0],car[1]] = 1
        for c in self._game.custs.values():
            # ignore served customers
            if not (c.served or c.hide):
                pos = c.pos
                # active customer
                if c.requested:
                    canvas[2,pos[0],pos[1]] = 1
                # potential customer not yet requesting
                else:
                    canvas[1,pos[0],pos[1]] = 1
        
        if not normalize:
            canvas = canvas * 255

        return canvas, [self._game.remaining_time / self._game.game_length]

    def _render_classic(self):
        """Provides a "classic" representation of the current game state.
        
        Note that it isn't purely classical, since some of the information returned is
        provided visually.

        Returns:
            dict: all potentially-relevant state information:
                - the car's location ("car"),
                - the depot's location ("depot"),
                - an array of the manhattan-grid with 1s at locations with current active customers ("curr_cust"),
                - same type of array for the potential customers ("potential_cust"),
                - the current time ("time"),
                - and the remaining time ("remaining_time").
        """
        
        # most info ready to send off, but we must first make the customer grid arrays
        
        # get list of current customer positions
        curr_cust_list = [c.pos for c in self._game.custs.values() if (c.requested and not (c.served or c.hide))]
        # split into list of x-coords and y-coords
        curr_cust_coords = ([e[0] for e in curr_cust_list],[e[1] for e in curr_cust_list])
        # turn into feature-layer-like grids
        curr_cust_grid = np.zeros(self.game_config['shape'])
        curr_cust_grid[curr_cust_coords] = 1
        # same process for potential customers
        potential_cust_list = [c.pos for c in self._game.custs.values() if (not c.requested and not (c.served or c.hide))]
        potential_cust_coords = ([e[0] for e in potential_cust_list],[e[1] for e in potential_cust_list])
        potential_cust_grid = np.zeros(self.game_config['shape'])
        potential_cust_grid[potential_cust_coords] = 1
        
        return {
            'car': [self._game.car_pos],
            'depot': [self._game.depot_pos],
            'curr_cust':curr_cust_grid,
            'potential_cust':potential_cust_grid,
            'time':[self._game.curr_time],
            'remaining_time':[self._game.remaining_time]
        }
