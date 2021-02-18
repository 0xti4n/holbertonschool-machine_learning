#!/usr/bin/env python3
"""Deep Q-learning train atari game"""

from PIL import Image
import numpy as np
import gym


from keras import layers
import keras as K
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Class atari processor"""

    def process_observation(self, observation):
        """Process observation"""
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """process state batch"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """process reward"""
        return np.clip(reward, -1., 1.)


def create_model(actions):
    """function that create the model for agent"""
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    inputs = K.Input(input_shape)
    model = layers.Permute((2, 3, 1))(inputs)

    model = layers.Conv2D(filters=32,
                          kernel_size=(8, 8),
                          strides=(4, 4),
                          activation='relu')(model)

    model = layers.Conv2D(filters=64,
                          kernel_size=(4, 4),
                          strides=(2, 2),
                          activation='relu')(model)

    model = layers.Conv2D(filters=64,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          activation='relu')(model)

    model = layers.Flatten()(model)
    model = layers.Dense(512, activation='relu')(model)
    output = layers.Dense(actions, activation='linear')(model)

    return K.Model(inputs=inputs, outputs=output)


if __name__ == '__main__':
    """main function where train the model agent"""
    np.random.seed(1)
    filename = 'policy.h5'
    log_filename = 'policy_log.json'
    checkpoint_w_file = 'dqn_weights_{step}.h5'
    env = gym.make('Breakout-v0')
    env.seed(1)
    env.reset()
    actions = env.action_space.n
    model = create_model(actions)
    memory = SequentialMemory(limit=1000000,
                              window_length=WINDOW_LENGTH)

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000)

    processor = AtariProcessor()
    dqn = DQNAgent(model, policy=policy, nb_actions=actions,
                   nb_steps_warmup=1000,
                   delta_clip=1.,
                   memory=memory,
                   train_interval=4,
                   processor=processor,
                   target_model_update=100)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    callbacks = [ModelIntervalCheckpoint(filepath=checkpoint_w_file,
                                         interval=10000,
                                         verbose=1)]

    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks,
            nb_steps=100000,
            visualize=True)

    dqn.save_weights(filename, overwrite=True)
