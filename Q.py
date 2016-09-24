# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:59:29 2014

@author: nima
"""
import numpy as np
from Env import *
Direction = {0: 'u', 1:'d', 2:'r', 3:'l'}
class QLanda(object):
    Count = 0
        
    #Q = np.random.rand(36,4)/10.0
    Q = np.zeros((36,4))
    #Q = np.float64(np.random.rand(36,4))
    e = np.zeros((36,4))
    Ns = np.zeros((36,4))
    Policy = []
    
    
    def __init__(self, iteration, gama, landa, alpha, epsilon):
        self.iteration = iteration
        self.gama = gama
        self.env = Environment()
        self.alpha = alpha
        self.landa = landa
        self.epsilon = epsilon
    def start(self):        
        for item in range(self.iteration):        
            s = np.random.randint(0,36)
            a = np.random.randint(0,4)
            #print "________________",self.Count
            self.learn(s,a)
            self.Count += 1 
    
    def learn(self,state,action):
        s = state
        a = action        
        gama = float(self.gama)
        landa = float(self.landa)
        while s != 5:              
            rnd = np.random.randint(1,100)            
            r= 0
            ns,a = self.env.get_next_state_undeterministic(s,a)            
            if ns == 5:
                r = 10
            aPrime = self.Q[ns].argmax()
            if rnd <= self.epsilon:
                aFinal = aPrime
            else:
                aFinal = np.random.randint(0,4)
            delta = r +  (gama* self.Q[ns][aPrime]) - self.Q[s][a]
            self.e[s][a]+= 1.0
            for cs in range(0,self.Q.shape[0]):
                for ca in range(0,self.Q.shape[1]):                    
                    self.Q[cs][ca] +=  0.1 * delta * self.e[cs][ca]
                    if aFinal == aPrime:
                        self.e[cs][ca] = (gama*landa) * self.e[cs][ca]
                    else:
                        self.e[cs][ca] = 0
            s = ns
            a = aFinal



    
    
class Q(object):
    Count = 0    
    #Q = np.random.rand(36,4)/10.0
    Q = np.zeros((36,4))
    Policy = []
    
    def __init__(self,iteration,landa):
        self.iteration = iteration
        self.landa = landa 
        self.env = Environment()
        
    
    def update_q(self,state,action):
        
        s = state
        a = action
        s1=0  
        s2=0 
        s3=0
        Qs1=0 
        Qs2=0 
        Qs2=0
        if a == 0 :           
            s1 = self.env.get_next_state(s,0)
            s2 = self.env.get_next_state(s,2)
            s3 = self.env.get_next_state(s,3)
            Qs1 = 0.8 * max([self.Q[s1][0],self.Q[s1][1],self.Q[s1][2],self.Q[s1][3]])
            Qs2 = 0.1 * max([self.Q[s2][0],self.Q[s2][1],self.Q[s2][2],self.Q[s2][3]])
            Qs3 = 0.1 * max([self.Q[s3][0],self.Q[s3][1],self.Q[s3][2],self.Q[s3][3]])
            
        elif a == 1 :           
            s1 = self.env.get_next_state(s,1)
            s2 = self.env.get_next_state(s,2)
            s3 = self.env.get_next_state(s,3)
            Qs1 = 0.8 * max([self.Q[s1][0],self.Q[s1][1],self.Q[s1][2],self.Q[s1][3]])
            Qs2 = 0.1 * max([self.Q[s2][0],self.Q[s2][1],self.Q[s2][2],self.Q[s2][3]])
            Qs3 = 0.1 * max([self.Q[s3][0],self.Q[s3][1],self.Q[s3][2],self.Q[s3][3]])
            
            
        elif a == 2 :           
            s1 = self.env.get_next_state(s,2)
            s2 = self.env.get_next_state(s,0)
            s3 = self.env.get_next_state(s,1)
            Qs1 = 0.8 * max([self.Q[s1][0],self.Q[s1][1],self.Q[s1][2],self.Q[s1][3]])
            Qs2 = 0.1 * max([self.Q[s2][0],self.Q[s2][1],self.Q[s2][2],self.Q[s2][3]])
            Qs3 = 0.1 * max([self.Q[s3][0],self.Q[s3][1],self.Q[s3][2],self.Q[s3][3]])
            
            
        elif a == 3 :           
            s1 = self.env.get_next_state(s,3)
            s2 = self.env.get_next_state(s,0)
            s3 = self.env.get_next_state(s,1)
            Qs1 = 0.8 * max([self.Q[s1][0],self.Q[s1][1],self.Q[s1][2],self.Q[s1][3]])
            Qs2 = 0.1 * max([self.Q[s2][0],self.Q[s2][1],self.Q[s2][2],self.Q[s2][3]])
            Qs3 = 0.1 * max([self.Q[s3][0],self.Q[s3][1],self.Q[s3][2],self.Q[s3][3]])
            
        #self.Q[s][a] = self.env.get_reward(s,a)+ self.landa* (Qs1 + Qs2 + Qs3)
        self.Q[s][a] = self.env.get_expected_reward(s,a)+ self.landa* (Qs1 + Qs2 + Qs3)
        
    def learn_q(self):
        count = 0
        
        while count <= self.iteration:
            for s in range(0,self.Q.shape[0]):
                for a in range(0,self.Q.shape[1]):
                    self.update_q(s,a)
            count +=1
            #print "____________iteration number : %s ___________"%(count)
            

