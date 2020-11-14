#!/usr/bin/env python


import unittest
import gym
import gym_brain  


class Environments(unittest.TestCase):
    def test_env(self):
        env = gym.make("Brain-v0")
        env.seed(0)
        env.reset()
        env.step(0)
