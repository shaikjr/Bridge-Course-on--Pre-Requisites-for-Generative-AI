# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:59:43 2023

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 20:20:14 2023

@author: Admin
"""

import gym

import gym_bandits

env = gym.make("BanditTwoArmedHighLowFixed-v0")
#print(env.action_space.n)
#print(env.p_dist)

import numpy as np
count = np.zeros(2)

sum_rewards = np.zeros(2)
Q = np.zeros(2)
num_rounds = 100

def UCB(i):
    ucb = np.zeros(2)
    if i < 2:
        return i
    else:
        for arm in range(2):
            ucb[arm] = Q[arm] + np.sqrt((2*np.log(sum(count))) / count[arm])
    return (np.argmax(ucb))
env.reset()

for i in range(num_rounds):
    arm = UCB(i)
    next_state, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm]+=reward
    Q[arm] = sum_rewards[arm]/count[arm]
  

print(Q)
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))