"""Sample wrappers for multi-agent SEED setup"""

import gym
import numpy


class SampleMultiAgentRewardWrapper(gym.RewardWrapper):
  """Converts reward for each player to one collective reward."""

  def __init__(self, env):
    super(SampleMultiAgentRewardWrapper, self).__init__(env)

  def reward(self, reward):
    return numpy.max(reward)


# Beware that this wrapper is probably not the best (information
# about players and ball is duplicated)
class SampleMultiAgentObservationWrapper(gym.ObservationWrapper):
  """Joins per-player observations into one."""

  def __init__(self, env):
    super(SampleMultiAgentObservationWrapper, self).__init__(env)
    observation_shape = env.observation_space.shape[1:-1] + (
      env.observation_space.shape[0] * env.observation_space.shape[-1],)
    self.observation_space = gym.spaces.Box(
      low=0,
      high=255,
      shape=observation_shape,
      dtype=numpy.uint8)

  def observation(self, observation):
    return numpy.concatenate(observation, axis=-1)
