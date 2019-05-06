import gym
env = gym.make("MountainCarContinuous-v0").unwrapped
print("actions: ", env.action_space) # 查看这个环境中可用的 action 有多少个
print("observations: ",  env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

#observation = env.reset()
#for _ in range(1000):
#  env.render()
#  action = env.action_space.sample() # your agent here (this takes random actions)
#  observation, reward, done, info = env.step(action)
#
#  if done:
#    observation = env.reset()
#env.close()
