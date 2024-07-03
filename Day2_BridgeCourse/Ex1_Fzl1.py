# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:33:37 2023

@author: HP
"""


import gymnasium as gym
import time
env = gym.make("FrozenLake-v1", render_mode = "human")

env.reset()
env.render() 
time.sleep(10)

#print the states
print(env.observation_space)
#print the action space left =0 down = 1, right = 2 and up=3
print(env.action_space)  

#print the transition probs of a state. P(s'|s,a). Gives the prob of moving to
#state s' from state s by performing action a. 
#syntax : P[state][action]
print(env.P[0][2])
# output is of the form : transn_prob, next_state, reward, is terminal state? 


#perform a deterministic action step 0,1,2,3
print("Deterministic Action:")
(next_state, reward, done, transn_prob,info) = env.step(2) 
#return_values are next_state, reward, terminal_state?, debug_info, 
#transn_prob of going back to the previous state due to this action
print("Next state is : {0}, Reward is {1}, Is a Terminal state: {2}".format(next_state, reward,done))
env.render()
time.sleep(10)
env.reset()

#generate a random action and perform it
print("Random Action:")
rnd_action = env.action_space.sample() 
(next_state, reward, done, transn_prob,info) = env.step(rnd_action)
print("Next state is : {0}, Reward is {1}, Is a Terminal state: {2}".format(next_state, reward,done))

env.render()
time.sleep(10)


#generate an episode with a random policy for 20 time-steps
env.reset()
num_timesteps = 20

for t in range(num_timesteps):
    rnd_action = env.action_space.sample()
    next_state, reward, done, info, transn_prob = env.step(rnd_action)
    print("Time:{0}\tNext:{1}\t Reward:{2}\t, Terminal?:{3}".format(t+1,next_state,reward,done))
    env.render()
    time.sleep(10)
    if done:
        break
env.close()
