# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:00:38 2014

@author: nima
"""
import numpy as np
from Env import *
import matplotlib.pyplot as plt
class TD(object):
        
    
        
    #Q = np.random.rand(36,4)/10.0
    #Q = np.zeros((36,4))
    utility = np.zeros((1,36))    
    Ns = np.zeros((1,36))
    checked = np.zeros((1,36))
    previous_state = None
    previous_action = None
    
    
    
    
    def __init__(self,policy,iteration,gama):
        self.iteration = iteration
        self.gama = gama
        self.policy = policy
        self.env = Environment()
        self.state0lst = np.zeros((1,iteration))
        self.state14lst = np.zeros((1,iteration))
        self.state28lst = np.zeros((1,iteration))
        
    def start(self): 
        print "start method"
        gama = self.gama
        for item in range(self.iteration):
            #for s in range(0,self.Q.shape[0]):
            s = np.random.randint(0,36)            
            #print "________________",item
            self.do_episod(s,gama)
            self.previous_action = None
            self.previous_state = None            
            self.state28lst[0][item]=self.utility[0][28]
            self.state14lst[0][item]=self.utility[0][14]
            self.state0lst[0][item]=self.utility[0][0]
            
    def do_episod(self,state,gama):
        s = state
        while s != None:
            if s == 5:
                r = 10
            else:
                r=0
            self.learn(s,r)
            na = self.previous_action
            s = self.previous_state
            if s == None:
                break
            else :
                s,a = self.env.get_next_state_undeterministic(s,na)
                #self.previous_action =a
    def learn(self,state,reward):
        
        gama = self.gama
        s = state
        r = reward        
        ps = self.previous_state
        pa = self.previous_action
        us = self.utility[0][s]
        ups = self.utility[0][ps]
        pr = 0
        
        if self.checked[0][s] == 0:
            self.utility[0][s]= r
        if ps != None:
            self.Ns[0][ps]+=1
            alpha = 1.0/ (self.Ns[0][ps])
            #alpha = 0.5
            self.utility[0][ps] += alpha*(pr + (gama*us) - ups )
        if s == 5 :
            self.previous_action = None
            self.previous_state =  None
        else :
            self.previous_action = self.policy[s].argmax()
            self.previous_state = s
        
    def visualize(self):
        
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.plot(range(self.iteration),self.state0lst[0],'-')#,fig.autofmt_xdate(),c='b')
        plt.title("Utility Diagram Using TD For state (1,1)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax2 = fig.add_subplot(222)
        ax2.plot(range(self.iteration),self.state14lst[0],'-')#,fig.autofmt_xdate(),c='r')
        plt.title("Utility Diagram Using TD For state (3,3)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax3 = fig.add_subplot(223)
        ax3.plot(range(self.iteration),self.state28lst[0],'-')#,fig.autofmt_xdate(),c='y')
        plt.title("Utility Diagram Using TD For state (5,5)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility') 
        plt.grid(True)
        
        plt.show()    
            
class TDLanda(object):
        
    
        
    #Q = np.random.rand(36,4)/10.0
    #Q = np.zeros((36,4))
    utility = np.zeros((1,36))    
    Ns = np.zeros((1,36))
    checked = np.zeros((1,36))
    e = np.zeros((1,36))
    current_state = None
    current_action = None
    
    
    
    
    def __init__(self,policy,iteration,gama,landa):
        self.iteration = iteration
        self.gama = gama
        self.policy = policy
        self.landa = landa
        self.env = Environment()
        self.state0lst = np.zeros((1,iteration))
        self.state14lst = np.zeros((1,iteration))
        self.state28lst = np.zeros((1,iteration))
        
    def start(self): 
        print "start method"
        gama = self.gama
        for item in range(self.iteration):
            #for s in range(0,self.Q.shape[0]):
            s = np.random.randint(0,36) 
            self.current_state = s
            #print "________________",item
            self.do_episod(s)
            self.previous_action = None
            self.previous_state = None            
            self.state28lst[0][item]=self.utility[0][28]
            self.state14lst[0][item]=self.utility[0][14]
            self.state0lst[0][item]=self.utility[0][0]
            
    def do_episod(self,state):
        s = state
        gama = self.gama
        landa = self.landa
        while s != 5:
            action = self.policy[s].argmax()
            ns ,a = self.env.get_next_state_undeterministic(s,action)
            if ns == 5:
                r=10
            else :
                r=0
                
            delta = r + (gama* self.utility[0][ns]) - self.utility[0][s]
            self.e[0][s]+=1
            for st in range(36):
                self.utility[0][st] +=  (0.2 * delta * self.e[0][st])
                self.e[0][st] = landa * gama * self.e[0][st]
            s = ns   
                
    
    def visualize(self):

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.plot(range(self.iteration),self.state0lst[0],'-')#,fig.autofmt_xdate(),c='b')
        plt.title("Utility Diagram Using TDLanda For state (1,1)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax2 = fig.add_subplot(222)
        ax2.plot(range(self.iteration),self.state14lst[0],'-')#,fig.autofmt_xdate(),c='r')
        plt.title("Utility Diagram Using TDLanda For state (3,3)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility')
        plt.grid(True)
        
        ax3 = fig.add_subplot(223)
        ax3.plot(range(self.iteration),self.state28lst[0],'-')#,fig.autofmt_xdate(),c='y')
        plt.title("Utility Diagram Using TDLanda For state (5,5)")
        plt.xlabel('Iteration')
        plt.ylabel('Utility') 
        plt.grid(True)
        
        
        
        plt.show()
        
        
        
        
     
        
        
       