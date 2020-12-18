# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 04:58:41 2019

@author: tarkesh2shar
"""

#Thompson Sampling!!!!!!!!!!!!!



import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
import random;
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")


#//whole algorithm 

d=10    # versions of the ads ok ?
N=10000
ads_selected=[];

numbers_or_rewards_1=[0] *d

numbers_or_rewards_0=[0] *d
total_reward=0;

for n in range(0,N):
    
    ad=0;
    max_random=0;
    
    for i in range(0,d):
        
        #take random draws 
        random_beta=random.betavariate(numbers_or_rewards_1[i]+1,numbers_or_rewards_0[i]+1)
      
        if random_beta>max_random:
           max_random=random_beta;
           ad=i
    ads_selected.append(ad)
 
    reward=dataset.values[n,ad];
    if reward==1:
        numbers_or_rewards_1[ad]=numbers_or_rewards_1[ad]+1
    else:
        numbers_or_rewards_0[ad]=numbers_or_rewards_0[ad]+1
    
    total_reward=total_reward+reward
    

plt.hist(ads_selected)
plt.title("ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each add was selected")
plt.show()