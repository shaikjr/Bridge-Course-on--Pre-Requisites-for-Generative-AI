# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 15:11:24 2023

@author: HP
"""

import gymnasium as gym 

#from gymnasium.wrappers.monitoring import video_recorder
#from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder 

env = gym.make("CartPole-v1", render_mode = "human")

env.reset()

env.render()

#print state space which is continuous
print(env.observation_space)

#print action space
print(env.action_space)


env.reset()
#implement with a random policy


n_episodes = 50
n_timesteps = 50

for i in range(n_episodes):
    
    Return = 0
    for t in range(n_timesteps):
        env.render()
       
        rnd_action = env.action_space.sample()
        next_state, reward, done, infor, prob = env.step(rnd_action)
        Return = Return + reward
        if done:
            env.reset()
            break
    if i%10 == 0:
        print("Episode : {}, Return : {}".format(i+1, Return))

env.close()