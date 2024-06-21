#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:48:13 2024

@author: claire
"""

import numpy as np
import random
import pygame

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.viewer import Viewer

class FixedViewer(Viewer):
    """
    A version of Viewer for mushroomrl that doesn't hang
    """
    def display(self, s):
        """
        Display current frame and initialize the next frame to the background
        color.

        Args:
            s: time to wait in visualization.

        """
        pygame.display.flip()
        
        #use pygame.time.delay not time.sleep
        pygame.time.delay(int(s*1000))
        #process pygame internal events e.g. close window buttons
        pygame.event.pump()

        self.screen.fill(self._background)



class Pong(Environment):
    """
    A simplified Pong environment.
    
    Parameters
    ----------
    horizon : int, default is 100
        DESCRIPTION. Horizon of the problem.
    gamma : float, default is .95
        DESCRIPTION. Discount factor
    dt : float, default is 0.1
        DESCRIPTION. time step

    Returns
    -------
    None.

    """
    def __init__(self, horizon=100, gamma=.95, dt=0.1):
        
        #MDP parameters
        #Lower bounds: x_min, y_min, vx_min, vy_min
        self.lower = np.array([0, 0, -1, -1])
        
        #upper bounds: x_max, y_max, vx_max, vy_max
        self.upper = np.array([1, 1, 1, 1])
        
        #create observation space
        observation_space = spaces.Box(low=self.lower, high=self.upper)
        
        #create action space
        n_actions = 5
        action_space = spaces.Discrete(n_actions)
        
        #create bins for paddle
        self.n_bins = n_actions
        bin_width = 1.0 / n_actions
        self.bins = np.linspace(bin_width, 1.0, n_actions, endpoint=True)

    
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)
        
        #Visualisation
        self._viewer = FixedViewer(1.2, 1.2)
        
        super().__init__(mdp_info)
        
    def reset(self, state=None):
        if state is None:
            x_start = random.uniform(0, 1)
            y_start = random.uniform(0, 1)
            vx_start = random.uniform(-1, 1)
            vy_start = random.uniform(-1, 1)
            state = np.array([x_start, y_start, vx_start, vy_start])
        self._state = state
        return state
    
    def step(self, action):
        """
        Update the state to the next time step given the action.
        """    
        x_min, y_min, vx_min, vy_min = self.lower
        x_max, y_max, vx_max, vy_max = self.upper
        x, y, vx, vy = self._state
        [paddle] = action
        
        # update state: increment x and y by v.dt
        x = x + vx * self.info.dt
        y = y + vy * self.info.dt
        
        # set default reward and absorbing
        reward = 0
        absorbing = False
        
        #if pong on left side, so x < x_min, just reflect
        if x < x_min:
            x = x_min
            vx = -vx
            
        #if pong is on right side, so x > x_max, just reflect
        if x > x_max:
            x = x_max
            vx = -vx   
            
        #if pong is on top side, so y > y_max, just reflect
        if y > y_max:
            y = y_max
            vy = -vy   
            
        #if pong is on bottom, so y < y_min
        if y < y_min:    
            # get y range that pong falls into
            bins = self.bins
            x_bin = np.digitize(x, bins)
            if x_bin == self.n_bins:
                x_bin = x_bin - 1
            # if paddle in postion, reflect, reward
            if paddle == x_bin:
                reward = 0.2
                y = y_min
                vy = -vy
            else:
                reward = -1
                absorbing = True
        # small reward if paddle returns to left, penalty otherwise
        else:
            if paddle == 0:
                reward = 0.1
            else:
                reward = -0.1
                
        self._state = np.array([x, y, vx, vy])
        self._paddle = paddle
        
        return self._state, reward, absorbing, {}
    
    def render(self, record=False):
        """ Display the box, the moving Pong and the moving paddle
        """
        position = [self._state[0]+0.1, self._state[1]+0.1]
 
        self._viewer.circle(position, 0.01, color=[255, 0, 0])
        self._viewer.line([0.1, 0.1], [0.1, 1.1], color=(0, 0, 255))
        self._viewer.line([0.1, 1.1], [1.1, 1.1], color=(0, 0, 255))
        self._viewer.line([1.1, 0.1], [1.1, 1.1], color=(0, 0, 255))
        
        x_start = self._paddle/self.n_bins + 0.1
        x_end = self._paddle/self.n_bins + 1/self.n_bins + 0.1
        
        self._viewer.line([x_start, 0.1], [x_end, 0.1], color=(0, 255, 255))
        
        frame = self._viewer.get_frame() if record else None

        self._viewer.display(self.info.dt)

        return frame

    def stop(self):
        self._viewer.close()
       
        
        
        
        
        
        