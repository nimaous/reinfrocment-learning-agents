# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 17:37:42 2014

@author: nima
CopyRight By Nima Rafiee . Rafiee.nima@gmail.com
you are free to extend this code for none commercial use 
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

index 0 => U
index 1 => D
index 2 => R
index 3 => L

[ 0 ,1 ,2 ,3 ,4 ,5
    --------
[ 6 ,7 ,8 ,9 ,10,|11
  12,13,14,15,16,|17
        --
  18,19,20|,21,22,23
  24,25,26|,27,28,29
  30,31,32,33,34,35]]


"""

import numpy as np

Direction = {0: 'u', 1:'d', 2:'r', 3:'l'}

class Environment(object):
    
    states_utilities = np.zeros((1,36))
    def get_expected_reward(self,state,action):
        s = state
        a = action        
        if a==0:
            up = self.get_reward(s,0)
            right = self.get_reward(s,2)
            left = self.get_reward(s,3)            
            return 0.8 * up + 0.1* (right + left)
        if a==1:
            down = self.get_reward(s,1)
            right = self.get_reward(s,2)
            left = self.get_reward(s,3)            
            return 0.8 * down + 0.1* (right + left)
        if a==2:        
            right = self.get_reward(s,2)
            down = self.get_reward(s,1)
            up = self.get_reward(s,0)            
            return 0.8 * right + 0.1* (up + down)
        if a==3:
            left = self.get_reward(s,3)
            down = self.get_reward(s,1)
            up = self.get_reward(s,0)            
            return 0.8 * left + 0.1* (up + down)
            
    def get_reward(self,state,action):
        s = state
        a = action
        
        if s==4 and a==2:
            return 10
        elif s==11 and a==0:
            return 10
        else:
            return 0
 
    def get_next_state(self,state,action):
        
        s = state
        #a = self.get_action(action)    
        a = action
        if s == 5 :
            return s 
        if a == 0 :        
            if s<=5 or 7 <= s <= 9 or s==20:
                return s
            else :
                return s-6
        
        elif a==1:
            if s>=30 or 1<= s <=3 or s==14 :
                return s
            else:
                return s+6
            
        
        elif a==2:
            if s%6 ==5 or s==10 or s==16 or s==20 or s==26:
                return s
            else :
                return s+1
        
        else :
            if s%6==0 or s==1 or s==17 or s==21 or s==27:
                return s
            else :
                return s-1
    def get_action(self,action):        
        rnd = np.random.randint(0,10)        
        if  action == 0 :
            if rnd <8:
                return action
            elif rnd ==8:
                return 2
            else :
                return 3
        elif action == 1 :
            if rnd < 8:
                return action
            elif rnd ==8:
                return 2
            else :
                return 3
        elif action == 2 :
            if rnd < 8:
                return action
            elif rnd ==8:
                return 0
            else :
                return 1
        else :
            if rnd < 8:
                return action
            elif rnd == 8:
                return 0
            else :
                return 1    
    def get_next_state_undeterministic(self,state,action):
        
        s = state
        a = self.get_action(action)    
        #a = action
        if s == 5 :
            return s,a 
        if a == 0 :        
            if s<=5 or 7 <= s <= 9 or s==20:
                return s,a
            else :
                return s-6,a
        
        elif a==1:
            if s>=30 or 1<= s <=3 or s==14 :
                return s,a
            else:
                return s+6,a
            
        
        elif a==2:
            if s%6 ==5 or s==10 or s==16 or s==20 or s==26:
                return s,a
            else :
                return s+1,a
        
        else :
            if s%6==0 or s==1 or s==17 or s==21 or s==27:
                return s,a
            else :
                return s-1,a
