#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Last name Nordlund => A=1

#%%

#%% Basic rules
# 0 right, 1 down, 2 left, 3 up
sp = np.array([
    [1,3,-1,-1], [2,4,0,-1], [-1,5,1,-1],
    [4,6,-1,0], [5,7,3,1], [-1,8,4,2],
    [7,-1,-1,3], [8,-1,7,5], [-1,-1,7,5]
    ]) # defines what state results from a step in each direction depending on start state

number_of_states = sp.shape[0] # rows
number_of_actions = sp.shape[1] # columns

def step(s, action):
    next_state = sp[s,action]
    reward =- 1
    
    # from rightmost states going right (states 2, 5, 8 action 0)
    if (s==2 and action==0) or (s==5 and action==0) or (s==8 and action==0):
        reward=-1+1
        
    # from leftmost states going left (states 0, 3, 6 action 2)
    if (s==0 and action==2) or (s==3 and action==2) or (s==6 and action==2):
        reward=-1-1
        
    # from bottom states going down (states 6, 7, 8 action 1)
    if (s==6 and action==1) or (s==7 and action==1) or (s==8 and action==1):
        reward=-1+1
        
    # from top states going up (states 0, 1, 2 action 3)
    if (s==0 and action==3) or (s==1 and action==3) or (s==2 and action==3):
        reward=-1+2
        
    return next_state, reward

#%% Q-table calculated by hand
Manual_Q = np.array([ # actions: right, down, left, up in order
    [0,-1,-2,1], [0,-1,0,1], [0,-1,0,1], 
    [-1,-1,-2,0], [-1,-1,-1,0], [0,-1,-1,0], 
    [-1,0,-2,-1], [-1,0,-1,-1], [0,0,-1,-1]    
    ])
Manual_Q_df = pd.DataFrame(data=Manual_Q,columns=['go right', 'go down', 'go left', 'go up'])

#%% Init Q-table
Q_values=np.full((9,4), 0.0)
Q_values_df=pd.DataFrame(data=Q_values,columns=['go right','go down', 'go left','go up'])

#%%

#%% Training in two steps
Q_values_temp = Q_values.copy()
for i in range(2):
    Q_values_temp=Q_values.copy()
    for s in range(number_of_states):
        for a in range(number_of_actions):
           next_state,reward=step(s,a)
           
           if next_state==-1:
               Q_values_temp[s,a]=reward
           else:
               Q_values_temp[s,a]=reward+np.max(Q_values[next_state])
                              
    Q_values_2step=Q_values_temp.copy()
Q_2step_df = pd.DataFrame(data=Q_values_2step,columns=['go right','go down', 'go left','go up'])
# Has a couple of errors compared to the manually calculated table

#%%

#%% Training block
for i in range(100):
    Q_values_temp=Q_values.copy()
    for s in range(number_of_states):
        for a in range(number_of_actions):
           next_state,reward=step(s,a)
           
           if next_state==-1:
               Q_values_temp[s,a]=reward
           else:
               Q_values_temp[s,a]=reward+np.max(Q_values[next_state])
                              
    Q_values=Q_values_temp.copy()

Q_values_df=pd.DataFrame(data=Q_values,columns=['go right','go down', 'go left','go up'])
# This table perfectly aligns with the one calculated by hand
        
    
