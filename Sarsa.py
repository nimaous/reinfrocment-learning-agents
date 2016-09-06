# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:01:32 2014

@author: nima
"""
import numpy as np
from Env import *
Direction = {0: 'u', 1:'d', 2:'r', 3:'l'}
class SarsaLanda(object):
    Count = 0
        
    #Sarsa = np.random.rand(36,4)/10.0
    #Sarsa = np.zeros((36,4))
    
    Sarsa = np.random.rand(36,4)
    e = np.zeros((36,4))
    Ns = np.zeros((36,4))
    Policy = []
    
    
    def __init__(self,iteration=100,gama=0.9,landa=0.5,alpha=0.1,epsilon=90):
        self.iteration = iteration
        self.gama = gama
        self.env = Environment()
        self.alpha = alpha
        self.landa = landa
        self.epsilon = epsilon
    def start(self):
        print "start method"
        for item in range(self.iteration):
            #for s in range(0,self.Sarsa.shape[0]):
            s = np.random.randint(0,36)
            a = np.random.randint(0,4)
            print "________________",self.Count
            self.learn(s,a)
            self.Count += 1 
    
    def learn(self,state,action):
        s = state
        a = action
        gama = float(self.gama)
        landa = float(self.landa)
        while s != 5:
            rnd = np.random.randint(1,100)            
            
            self.Ns[s][a] +=1
            #print "s:",s
            #print "a:",a
            ns ,a= self.env.get_next_state_undeterministic(s,a)
            #print ns
            #print "ns:",ns
            r=0
            if ns == 5:
                r = 10
            aPrime = self.Sarsa[ns].argmax()
            if rnd <= self.epsilon:
                aFinal = aPrime
            else:
                aFinal = np.random.randint(0,4)
            delta = r +  (self.gama* self.Sarsa[ns][aFinal]) - self.Sarsa[s][a]
            #self.e[s][a]+= 1
            self.e[s][a]+= 1.0
            for cs in range(0,self.Sarsa.shape[0]):
                for ca in range(0,self.Sarsa.shape[1]):
                    #self.Sarsa[cs][ca] +=  (1.0/ float(self.Ns[s][a]))* delta * self.e[s][a]
                    self.Sarsa[cs][ca] +=  0.5* delta * self.e[cs][ca]
                    #if aFinal == aPrime:
                    self.e[cs][ca] = (gama*landa) * self.e[cs][ca]
                    #else:
                     #   self.e[s][a] = 0
            s = ns
            a = aFinal
            
        #self.learn(ns,aPrime)
    def get_action(self,state):        
        return np.argmax(self.Sarsa[state])            
