#!/usr/bin/env python3
"""agent play"""
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """has the trained agent play an episode:

    Args:
    -> env is the FrozenLakeEnv instance
    -> Q is a numpy.ndarray containing the Q-table
    -> max_steps is the maximum number of steps in the episode

    Returns: the total rewards for the episode
    """
    env.reset()
    env.render()
    state = 0
    done = False
    for step in range(max_steps):
        A = epsilon_greedy(Q, state, 0)
        new_state, reward, done, _ = env.step(A)
        env.render()
        state = new_state

        if done is True:
            return reward
    env.close()
