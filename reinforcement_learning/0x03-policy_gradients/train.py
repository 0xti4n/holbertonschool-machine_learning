#!/usr/bin/env python3
"""
TRain cartpole game
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train agent cartpole game

    Args:
    -> env: initial environment
    -> nb_episodes: number of episodes used for training
    -> alpha: the learning rate
    -> gamma: the discount factor
    -> show_result: ender the environment every 1000 episodes computed

    Return:
    all values of the score (sum of all rewards during one episode loop)
    """
    w = np.random.rand(4, 2)
    all_scores = []
    for ep in range(nb_episodes+1):
        state = env.reset()[None, :]

        rewards = []
        score = 0
        grads = []

        while True:
            if show_result and (ep % 1000 == 0):
                env.render()

            action, grad = policy_gradient(state, w)
            new_st, reward, done, _ = env.step(action)
            new_st = new_st[None, :]

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = new_st

            if done:
                break

        for i in range(len(grads)):
            discounts = sum([r * gamma**r for r in rewards[i:]])
            w += alpha * grads[i] * discounts

        all_scores.append(score)
        print("Ep: {}, Score: {}".format(ep, score), end='\r', flush=False)

    return all_scores
