# -*- coding: utf-8 -*-

#https://github.com/JKCooper2/gym-bandits.git
#cd
#pip install -e .
import numpy as np 
import gym_bandits 
import gym
from gym import wrappers


env = gym.make("BanditTenArmedGaussian-v0") 
env = wrappers.Monitor(env,'C:\DATASCIENCES')

env.action_space


#initialisez les variables:


# nombre de tours (itérations)
num_rounds = 20000


# Nombre de fois qu'un bras a été tiré
count = np.zeros(10)


# Somme des récompenses de chaque bras
sum_rewards = np.zeros(10)



# Q valeur qui est la récompense moyenne
Q = np.zeros(10)


#Maintenant nous définissons la fonction softmax_Boltzmann :

import math
import random

def softmax_Boltzmann(to):
    total = sum([math.exp(val/to) for val in Q]) 
    probs = [math.exp(val/to)/total for val in Q]
    #probabilité de la distribution de Boltzmann
    threshold = random.random()
    cumulative_prob = 0.0
    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if (cumulative_prob > threshold):
            return i
    return np.argmax(probs)



for i in range(num_rounds):
    # Sélectionnez le bras avec softmax_Boltzmann
    arm = softmax_Boltzmann(0.5)
    
    # Obtenez la récompense
    env.reset() 
    observation, reward, done, info = env.step(arm)
    
    # mettre à jour le compte de ce bras
    count[arm] += 1
    # Somme les récompenses obtenues du bras
    sum_rewards[arm]+=reward
    # calcule la valeur Q qui correspond aux récompenses moyennes du bras
    Q[arm] = sum_rewards[arm]/count[arm]
    
print( 'Le bras optimal est  {}'.format(np.argmax(Q)))

env.close
    




























