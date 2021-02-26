#!/usr/bin/env python3
"""SARSA(λ) algorithm"""

import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the
    next action
    Args:
    -> Q is a numpy.ndarray containing the q-table
    -> state is the current state
    -> epsilon is the epsilon to use for the calculation
    Returns: the next action index
    """
    p = np.random.uniform(low=0.0, high=1.0)
    if p < epsilon:
        A = np.random.randint(low=0.0, high=len(Q[state]))
    else:
        A = np.argmax(Q[state])

    return A


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs SARSA(λ):

    Args:
    -> env is the openAI environment instance
    -> Q is a numpy.ndarray of shape (s,a) containing the Q table
    -> lambtha is the eligibility trace factor
    -> episodes is the total number of episodes to train over
    -> max_steps is the maximum number of steps per episode
    -> alpha is the learning rate
    -> gamma is the discount rate
    -> epsilon is the initial threshold for epsilon greedy
    -> min_epsilon is the minimum value that epsilon should decay to
    -> epsilon_decay is the decay rate for updating
        epsilon between episodes

    Returns:
    -> Q, the updated Q table
    """
    obv = env.observation_space.n
    env_r = env.desc.reshape(obv)
    init_ep = epsilon
    elegibility = np.zeros(Q.shape)
    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for _ in range(max_steps):
            new_st, reward, done, _ = env.step(action)
            new_ac = epsilon_greedy(Q, new_st, epsilon)

            elegibility *= lambtha * gamma
            elegibility[state, action] += 1.0

            s_error = reward + gamma * Q[new_st][new_ac] - Q[state][action]
            Q[state][action] += alpha * s_error * elegibility[state][action]

            if done:
                break

            state = new_st
            action = new_ac
        exp = np.exp(-epsilon_decay * ep)
        epsilon = (min_epsilon + (init_ep - min_epsilon) * exp)
    return Q
