
import gymnasium as gym
import time

env = gym.make('FrozenLake-v1',render_mode = 'human' )
env.reset()
env.render()

#print state space
print(env.observation_space)

#print action space
print(env.action_space)

print(env.P[0][2])

env.reset()
(next_state, reward, done, trans_prob,info) = env.step(2) #Deterministic Action
env.render()
rnd_action = env.action_space.sample()## Random Action
env.step(rnd_action)
env.render()
print("Action taken:", rnd_action)
ret = 0

#generate 1 episode

for  e in range(10):
   num_timesteps = 20
   ret = 0
   a=[]
   env.reset()
   for t in range(num_timesteps):
     rnd_action = env.action_space.sample()
     (next_state, reward, done, trans_probability,info) = env.step(rnd_action)
    
 if rnd_action==0:
         a.append("left")
     elif rnd_action==1:
         a.append("down")
     elif rnd_action==2:
         a.append("right")
     else:
         a.append("up")   
     ret = ret + reward #calculate & print return of the episode
     env.render()
     time.sleep(2)
     if done:
         print("The return of this episode is ",ret)
         print(a)  
         break
env.close()