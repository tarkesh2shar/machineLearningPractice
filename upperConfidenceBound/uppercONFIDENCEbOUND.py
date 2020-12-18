import pandas as pd;
import matplotlib.pyplot as plt;
import numpy as np
import math

dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

N=10000
d=10

ads_selected=[]

numbers_of_selctions=[0]*d;

sums_of_rewards=[0]*d;




for n in range(0,N):
    ad=0
    max_upper_bound=0
    for i in range(0,d):
        average_reward=sums_of_rewards[i]/numbers_of_selctions[i]
        delta_i=math.sqrt(3/2 * math.log(n+1)/ numbers_of_selctions[i])
        upper_bound=average_reward+delta_i
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
        
        
        


