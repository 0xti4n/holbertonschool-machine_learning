#!/usr/bin/env python3
"""Monte carlo algorithm"""

import numpy as np
import gym


def gen_ep(env, policy, max_steps):
    """generate episode for an action

    Args:
    -> env: environment instance
    -> policy: is a function that takes in a state and returns
        the next action to take
    -> max_steps: is the maximum number of steps per episode

    Return: the episode
    """
    states = []
    rewards = []
    state = env.reset()
    obv = env.observation_space.n
    env_r = env.desc.reshape(obv)
    for _ in range(max_steps):
        action = policy(state)
        new_st, reward, done, _ = env.step(action)
        states.append(state)
        if env_r[new_st] == b'H':
            rewards.append(-1.0)
            return [states, rewards]

        if env_r[new_st] == b'G':
            rewards.append(1.0)
            return [states, rewards]

        rewards.append(reward)

        if done:
            break
        state = new_st

    return [states, rewards]


def monte_carlo(env, V, policy, episodes=5000, max_steps=200,
                alpha=0.1, gamma=0.99):
    """performs the Monte Carlo algorithm:

    Args:
    -> env is the openAI environment instance
    -> V is a numpy.ndarray of shape (s,) containing the value estimate
    -> policy is a function that takes in a state and returns
        the next action to take
    -> episodes is the total number of episodes to train over
    -> max_steps is the maximum number of steps per episode
    -> alpha is the learning rate
    -> gamma is the discount rate

    Returns:
    -> V, the updated value estimate
    """
    discounts = [gamma**i for i in range(max_steps)]

    for _ in range(episodes):
        episode = gen_ep(env, policy, max_steps)
        G = 0
        for i in range(len(episode[0])):
            St = episode[0]
            r = episode[1]
            reward = np.array(r[i:])
            discount = np.array(discounts[:len(r[i:])])
            G = sum(reward * discount)
            V[St[i]] += alpha * (G - V[St[i]])
    return V
