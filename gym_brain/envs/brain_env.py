#!/usr/bin/env python

"""
Simulate the brain graph simulation environment.

Each episode is the evoluation of brain network for a patient.
"""

# Core Library
import math
import random
import numpy as np

import gym
from gym.spaces import Dict, Discrete, Box
from gym.utils import seeding

from graph_tool import Graph
from graph_tool.all import *
from graph_tool.spectral import laplacian

gamma_I = 0.2
gamma_X = 0.4
beta = 0.1
alpha_1 = 0.1
alpha_2 = 0.3

    
class BrainEnv(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, network_size=3, num_edges=2, directed=False):        
        self.network_size = network_size
        self.max_time_steps = 10
        
        #create the graph
        self.graph = Graph(directed=directed)
        self.graph.set_fast_edge_removal(True)
        self.graph.add_vertex(self.network_size)
        self.graph.add_edge(self.graph.vertex(0), self.graph.vertex(1))
        self.graph.add_edge(self.graph.vertex(1), self.graph.vertex(2))
        
        #associating load variable with graph nodes
        node_load = self.graph.new_vertex_property("double")
        node_load.get_array()[:] = np.random.random(network_size)
        
        amyloid = self.graph.new_vertex_property("double")
        node_load.get_array()[:] = np.random.random(network_size)
        
        self.graph.vertex_properties["load"] = node_load
        self.graph.vertex_properties["amyloid"] = amyloid
        
        edge_load = self.graph.new_edge_property("double")
        edge_load.get_array()[:] = np.random.random(num_edges)
        self.graph.edge_properties["load"] = edge_load
        
        #defining the action and state spaces
        action_low = np.zeros(network_size)
        action_high = np.ones(network_size)

        self.action_space = Box(action_low, action_high, dtype=np.float32)
        #self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        #self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        self.observation_space = Box(action_low, action_high, dtype=np.float32)
        
        #assigning load value to graph state
        self.cognition = 10
        self.time_step = 0
        self.observation = node_load #adjacency(self.graph).toarray().astype(int)
        self.A = adjacency(self.graph).toarray().astype(int)
        self.H = laplacian(self.graph).toarray().astype(int)
        self.seed_value = self.seed()
        self.curr_step = -1
        self.curr_episode = -1
        self.reset()
        
    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
        """
        done = 0
        reward = 0
        
        # include action checks and assertions here
        
        #compute the new load on each node
        self.graph.vertex_property['load'] += action 
        Iv = self.graph.vertex_property['load']
        
        #compute the information flow (I_v1v2)
        Iv1_v2 = calc_information_flow(self.graph)
        
        #update health of each node (X_v)
        Xv += _dXv_dt(self.graph)
        
        #compute energy consumed by each node
        Yv = calc_node_energy(Iv,Xv)
        
        #update health of each edge
        '''write logic here'''
        Xe = _dXe_dt(self.graph, Yv)
        
        #compute metabolism
        M = calc_edge_energy(Iv1_v2,Xe) + calc_node_energy(Iv,Xv)
        
        #compute reward
        Ct = np.sum(Iv)
        reward = self.cognition - Ct + M
        
        if self.curr_step == self.max_time_steps:
            done = 1
            
        self.observation = Iv
            
        return Iv, reward, done, []       

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation: List[int]
            The initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.graph.clear_edges()
        self.time_step = 0
        self.observation = self.graph.vertex_properties["load"]
        return self.observation

    def _render(self, mode = "graph"):
        if mode == 'graph':
            return self.graph
        elif mode == 'human':
            '''
            code for viewing the graph
            '''
            pass
            

    def _get_state(self):
        """Get the observation."""
        return self.observation

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def calc_node_energy(Iv, Xv):
        """compute node energy"""
        return np.sum(gamma_I*Iv) + np.sum(gamma_X/Xv)

    def calc_edge_energy(Iv1_v2, Xe):
        """compute edge energy"""
        return np.sum(gamma_I*Iv1_v2) + np.sum(gamma_X/Xe) 

    def calc_information_flow(G):
        '''
        define logic for computing information flow
        currently assuming information flow is average of load on edge vertices
        '''
        
        Iv1_v2 = np.zeros(len(G.edges()))
        for i,e in enumerate(G.edges()):
            Iv1_v2[i] = 0.5*G.vp.load[e.source()] + 0.5*G.vp.load[e.target] #taking average of Iv1 and Iv2
        return Iv1_v2

    def _dD_dt(G):
        H = laplacian(G)
        D = G.vp.amyloid
        assert H.shape[1] == D.shape[0]
        return -beta*H.dot(D)

    def _dXv_dt(G, Yv):
        #update the amyloid deposit
        D = G.vp.amyloid + _dD_dt(G)
        #compute change in health
        dXv_dt = -alpha_1*D - alpha_2*Yv
        return dXv_dt

