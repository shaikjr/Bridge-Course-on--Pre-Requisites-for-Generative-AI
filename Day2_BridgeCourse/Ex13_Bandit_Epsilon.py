# -*- coding: utf-8 -*-

import gym

import gym_bandits

env = gym.make("BanditTwoArmedHighLowFixed-v0")
print(env.action_space.n)
print(env.p_dist)

import numpy as np
count = np.zeros(2)

sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def epsilon_greedy(epsilon):
 
 if np.random.uniform(0,1) < epsilon:
     return env.action_space.sample()
 else:
     return np.argmax(Q)
 
env.reset()
for i in range(num_rounds):
    arm = epsilon_greedy(epsilon=0.5)
    next_state, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm]+=reward
    Q[arm] = sum_rewards[arm]/count[arm]

print(Q)
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))