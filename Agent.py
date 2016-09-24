# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 12:04:34 2014

@author: nima
"""
import numpy as np
import matplotlib.pyplot as plt
from ADP import *
from Q import *
from TD import *
from Sarsa import *
from Env import  *
#from Env import * 
Direction = {0: 'u', 1:'d', 2:'r', 3:'l'}


class Agent(object):
    

    
    
    def __init__(self):
        #self.init_state = initila_state
        self.policy = None
        self.env = Environment()
        self.flage = 1
            
    def q_learn(self,iteration,gama):
        self.policy = None
        self.Q = Q(iteration,gama)
        self.Q.learn_q()
        self.policy = self.Q.Q
        self.flage = 1
        
    def qlanda_learn(self,iteration,gama,landa,alpha=0.1,epsilon=90):
        self.policy = None
        self.QLanda = QLanda(iteration,gama,landa,alpha=0.1,epsilon=90)
        self.QLanda.start()
        self.policy = self.QLanda.Q
        self.flage = 1
        
    def sarsa_learn(self,iteration,gama,landa,alpha=0.1,epsilon=90):
        self.policy = None
        self.SarsaLanda = SarsaLanda(iteration,gama,landa,alpha,epsilon)
        self.SarsaLanda.start()
        self.policy = self.SarsaLanda.Sarsa
        self.flage = 1
            
    def show_policy(self):        
        pi=self.policy
        i=0
        while i <= 35 :  
            a0=pi[i].argmax()
            a1=pi[i+1].argmax()
            a2=pi[i+2].argmax()
            a3=pi[i+3].argmax()
            a4=pi[i+4].argmax()
            a5=pi[i+5].argmax()
            print Direction[a0],Direction[a1],Direction[a2],Direction[a3],Direction[a4],Direction[a5]     
            i += 6           
            print "\n"


    def adp_learn(self,iteration,gama):
        
        self.ADP = ADPLearn(self.policy,iteration,gama)
        self.ADP.start()
        self.ADP.visualize()        

        
        i = 0
        print "Utility Matrix"
        while i <= 35 :           
            print self.ADP.utility[0][i],'\t',self.ADP.utility[0][i+1],'\t',self.ADP.utility[0][i+2],'\t',self.ADP.utility[0][i+3],'\t',self.ADP.utility[0][i+4],'\t',self.ADP.utility[0][i+5]        
            i += 6
            print "\n"
                                                             
            
    def td_learn(self,iteration,gama):
        self.TD = TD(self.policy,iteration,gama)
        self.TD.start()
        self.TD.visualize()
        td = self.TD
        i = 0
        print "Utility Matrix"
        while i <= 35:
            
            print td.utility[0][i],'\t',td.utility[0][i+1],'\t',td.utility[0][i+2],'\t',td.utility[0][i+3],'\t',td.utility[0][i+4],'\t',td.utility[0][i+5]        
            i += 6
            print "\n"
            
    def tdLanda_learn(self,iteration,gama,landa):
        self.TD = TDLanda(self.policy,iteration,gama,landa)
        self.TD.start()
        self.TD.visualize()
        td = self.TD
        i = 0
        print "Utility Matrix"
        while i <= 35:
            
            print td.utility[0][i],'\t',td.utility[0][i+1],'\t',td.utility[0][i+2],'\t',td.utility[0][i+3],'\t',td.utility[0][i+4],'\t',td.utility[0][i+5]        
            i += 6
            print "\n"
            
    def use_policy(self,initial_state):
        
        state = initial_state
        policy = self.policy
        lst = []
        if state == 5 :            

            lst.append('Final Goal')
            print lst

            return 
        else :
            if self.flage :
                print """
                    [ 0 ,1 ,2 ,3 ,4 ,5
                        --------
                    [ 6 ,7 ,8 ,9 ,10,|11
                      12,13,14,15,16,|17
                            --
                      18,19,20|,21,22,23
                      24,25,26|,27,28,29
                      30,31,32,33,34,35]]
                             """
                self.flage = 0
                
            action = np.argmax(policy[state])            
            s1,a = self.env.get_next_state_undeterministic(state,action)
            lst.append(Direction[a])
            print "agent goes %s and State will be S%s"%(Direction[a],s1)        
            self.use_policy(s1)
        
        
                
                
if __name__ == '__main__':
    
    #iteration = input("please inter iteration)
    #agent = Qlearn(200,0.9)
    agent = Agent()
    #agent.q_learn(200,0.5)
    agent.qlanda_learn(200,0.9,0.5)
    #agent.td_learn(200,0.9)
    agent.tdLanda_learn(200,0.9,0.9)
   
    #agent.show_Q()
    
    #agent.adp_learn(100,0.9)    
    
    
    #obj.get_policy(26)    

    
   
    print "The END"
    

    
