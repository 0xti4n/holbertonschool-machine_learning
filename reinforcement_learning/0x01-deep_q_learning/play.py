#!/usr/bin/env python3
"""load model and play atari game"""

import gym

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

AtariProcessor = __import__('train').AtariProcessor
create_model = __import__('train').create_model

WINDOW_LENGTH = 4

if __name__ == '__main__':
    """play agent main function"""

    filename = 'policy.h5'
    env = gym.make('Breakout-v0')
    env.seed(1)
    env.reset()
    actions = env.action_space.n
    model = create_model(actions)
    memory = SequentialMemory(limit=1000000,
                              window_length=WINDOW_LENGTH)

    test_policy = GreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(model, nb_actions=actions,
                   test_policy=test_policy,
                   memory=memory,
                   processor=processor)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.load_weights(filename)
    dqn.test(env, nb_episodes=10, visualize=True)
