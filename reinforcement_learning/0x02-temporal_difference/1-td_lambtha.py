#!/usr/bin/env python3
"""TD(λ) lambtha"""

import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """performs the TD(λ) algorithm:

    Args:
    -> env is the openAI environment instance
    -> V is a numpy.ndarray of shape (s,) containing
        the value estimate
    -> policy is a function that takes in a state and
        returns the next action to take
    -> lambtha is the eligibility trace factor
    -> episodes is the total number of episodes to train over
    -> max_steps is the maximum number of steps per episode
    -> alpha is the learning rate
    -> gamma is the discount rate

    Returns:
    -> V, the updated value estimate
    """
    obv = env.observation_space.n
    eligibility = np.zeros(obv)
    env_r = env.desc.reshape(obv)

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = policy(state)
            new_st, reward, done, _ = env.step(action)

            eligibility *= lambtha * gamma
            eligibility[state] += 1.0

            if env_r[new_st] == b'G':
                reward = 1.0

            if env_r[new_st] == b'H':
                reward = -1.0

            td_error = reward + gamma * V[new_st] - V[state]

            V[state] += alpha * td_error * eligibility[state]

            if done:
                break

            state = new_st
    return V
