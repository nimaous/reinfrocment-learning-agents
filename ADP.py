# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:00:54 2014

@author: nima
"""
import numpy as np
from Env import *
import matplotlib.pyplot as plt

class ADPLearn(object):
    checked_sates = np.zeros((1,36))
    utility = np.zeros((1,36))
    Nsa = np.zeros((36,4))
    Ns1sa = np.zeros((36,36,4))
    Ps1sa = np.zeros((36,36,4))
    previous_state = None
    previous_action = None
    #state0lst  =[]
    #state14lst =[]
    #state28lst =[]
    plicy_evaluation_iteration = 50

    def __init__(self,policy,iteration,gama):
        self.iteration = iteration
        self.gama = gama
        self.policy = policy
        self.env = Environment()
        self.state0lst = np.zeros((1,iteration))
        self.state14lst = np.zeros((1,iteration))
        self.state28lst = np.zeros((1,iteration))
        #self.state2lst = np.zeros((1,iteration))
        
 
    def start(self): 
        
        for item in range(self.iteration):
            s = np.random.randint(0,36)            
            print "________________",item
            self.previous_action = None
            self.previous_state = None
            self.do_episod(s)            
            self.state28lst[0][item]=self.utility[0][28]
            self.state14lst[0][item]=self.utility[0][14]
            self.state0lst[0][item]=self.utility[0][0]
            #self.state2lst[0][item]=self.utility[0][34]
            

            
            
    def do_episod(self,state):
        s = state                         
        while s != None:            
            self.learn(s)
            na = self.previous_action  
            s = self.previous_state
            if s == None:
                break
                
            else:                
                s,a = self.env.get_next_state_undeterministic(s,na)                            
            

    def learn(self,state):
        reward = 0
        if state ==5:
            reward = 10
        ns = state
        ps = self.previous_state
        pa = self.previous_action        
        #print ns
        if self.checked_sates[0][ns] == 0 :
            self.utility[0][ns]=reward            
            self.checked_sates[0][ns]=1
        if ps != None:
            self.Nsa[ps][pa]+=1
            self.Ns1sa[ns][ps][pa] +=1
            for item in range(self.Ns1sa.shape[0]):
                if self.Ns1sa[item][ps][pa] != 0:
                    self.Ps1sa[item][ps][pa]= float(self.Ns1sa[item][ps][pa])/float(self.Nsa[ps][pa])                    
        self.policy_evaluation()

        if ns == 5 :
            self.previous_action = None
            self.previous_state = None
        else :
            self.previous_state = ns
            self.previous_action = self.policy[ns].argmax()
                    
    def policy_evaluation(self):        
        gama = self.gama
        reward = 0
        for state in range(36):            
            if state == 5:
                reward = 10
            else:
                reward = 0
            s1 = self.env.get_next_state(state,0)
            us1 = self.Ps1sa[s1][state][0]*self.utility[0][s1]
            s2 = self.env.get_next_state(state,1)
            us2 = self.Ps1sa[s2][state][1]*self.utility[0][s2]
            s3 = self.env.get_next_state(state,2)
            us3 = self.Ps1sa[s3][state][2]*self.utility[0][s3]
            s4 = self.env.get_next_state(state,3)
            us4 = self.Ps1sa[s4][state][3]*self.utility[0][s4]            
            self.utility[0][state] = reward +  (gama * (us1+us2+us3+us4))

        
        
                
             
    def visualize(self):
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(221)
        ax1.plot(range(self.iteration),self.state0lst[0],'-',fig1.autofmt_xdate(),c='b')
        plt.title("Utility Diagram Using ADP For state (1,1)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax2 = fig1.add_subplot(222)
        ax2.plot(range(self.iteration),self.state14lst[0],'-',fig1.autofmt_xdate(),c='r')
        plt.title("Utility Diagram Using ADP For state (3,3)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax3 = fig1.add_subplot(223)
        ax3.plot(range(self.iteration),self.state28lst[0],'-',fig1.autofmt_xdate(),c='y')
        plt.title("Utility Diagram Using ADP For state (5,5)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility') 
        plt.grid(True)
        
        plt.show()
        