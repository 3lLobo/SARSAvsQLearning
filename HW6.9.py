#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

def argmax(l):
    m = max(l)
    posmax = [i for i, j in enumerate(l) if j == m]
    
    return random.choice(posmax)
    
class SARSAQL:
    def __init__(self, eps=0.1):
        self.gamma = 1.0
        self.epsilon = eps
        self.alpha = 0.5
        self.optimalPath = None
        self.qvalues = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
               [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    
    ## Initialize policy
    #cell = [1/4,1/4,1/4,1/4]
    #row = []
    #for i in range(10):
    #    row.append(cell)

    #policy = []
    #for i in range(6):
    #    policy.append(row)
    
    #########################
    ## Initialize qvalues
    
    
#    for i in range(6):
#        for j in range(10):
#            for k in range(4):
#                qvalues[i][j][k] = random.random()
#    
#    for i in range(6):
#        qvalues[i][0][3] = -1000
#        qvalues[i][9][1] = -1000
#    
#    for i in range(10):
#        qvalues[0][i][0] = -1000
#        qvalues[5][i][2] = -1000
    ########################
    
    def sarsa(self, x, y):        
        action = self.policy((x,y))
        
        if action == 0:
            xynew = (x, y-1)
        elif action == 1:
            xynew = (x+1, y)
        elif action == 2:
            xynew = (x, y+1)
        elif action == 3:
            xynew = (x-1, y)

            
        orig_q = self.qvalues[y][x][action]
            
        if xynew[0] == 9 and xynew[1] == 5:
            self.qvalues[y][x][action] = orig_q + self.alpha * (10 - orig_q)
            return
        if xynew[1] == 5 and xynew[0] != 0:
            self.qvalues[y][x][action] = orig_q + self.alpha * (-100 - orig_q)
            return
        
        if xynew[0] < 0 or xynew[0] > 9:
            qnext = self.qvalues[y][x][self.policy((x,y))]
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.sarsa(x, y)
            return
        
        elif xynew[1] < 0 or xynew[1] > 5:
            qnext = self.qvalues[y][x][self.policy((x,y))]
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.sarsa(x, y)
            return
        
        else:    
            qnext = self.qvalues[xynew[1]][xynew[0]][self.policy(xynew)]
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.sarsa(xynew[0], xynew[1])
            
            return
        
    def qlearning(self, x, y):        
        action = self.policy((x,y))
        
        if action == 0:
            xynew = (x, y-1)
        elif action == 1:
            xynew = (x+1, y)
        elif action == 2:
            xynew = (x, y+1)
        elif action == 3:
            xynew = (x-1, y)

            
        orig_q = self.qvalues[y][x][action]
            
        if xynew[0] == 9 and xynew[1] == 5:
            self.qvalues[y][x][action] = orig_q + self.alpha * (10 - orig_q)
            return
        if xynew[1] == 5 and xynew[0] != 0:
            self.qvalues[y][x][action] = orig_q + self.alpha * (-100 - orig_q)
            return
        
        if xynew[0] < 0 or xynew[0] > 9:
            qnext = max(self.qvalues[y][x])
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.qlearning(x, y)
            return
        
        elif xynew[1] < 0 or xynew[1] > 5:
            qnext = max(self.qvalues[y][x])
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.qlearning(x, y)
            return
        
        else:    
            qnext = max(self.qvalues[xynew[1]][xynew[0]])
            self.qvalues[y][x][action] = orig_q + self.alpha * (-1 + self.gamma * qnext - orig_q)
            self.qlearning(xynew[0], xynew[1])
            
            return
            
    def policy(self, xy):
        x, y = xy
        rnd = random.random()
        
        if rnd < self.epsilon:
            actions = range(4)
                
            return random.choice(actions)
        else:
            return argmax(self.qvalues[y][x])
        

    def episode_sarsa(self):
        x = 0
        y = 5
        self.sarsa(x, y)
        
        return

    def episode_qlearning(self):
        x = 0
        y = 5
        self.qlearning(x, y)
        
        return
                
    def epsilon_greedy_sarsa(self, x):
        for i in range(x):
            self.episode_sarsa()
    
    def epsilon_greedy_qlearning(self, x):
        for i in range(x):
            self.episode_qlearning()
    
    def optimal_path(self, li):
        x, y = li[-1]
        if x == 9 and y == 5:
            self.optimalPath = li
            
            return li
        else:
            optimal = argmax(self.qvalues[y][x])
            
            if optimal == 0:
                xynew = (x, y-1)
            elif optimal == 1:
                xynew = (x+1, y)
            elif optimal == 2:
                xynew = (x, y+1)
            elif optimal == 3:
                xynew = (x-1, y)
            
            li.append(xynew)
            self.optimal_path(li)
            
        
    def show_optimal(self):
        grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 2, 2, 2, 2, 2, 0]]
        
        self.optimal_path([(0,5)])
        li = self.optimalPath
        
        for li in li:
            grid[li[1]][li[0]] = 1
            
        for item in grid:
            print(*item)
        return 
            
#testsarsa = SARSAQL()
#testsarsa.epsilon_greedy_sarsa(10000)
#print("SARSA optimal path:")
#testsarsa.show_optimal()

#testqlearning = SARSAQL()
#testqlearning.epsilon_greedy_qlearning(10000)
#print("\nQ-Learning optimal path:")
#testqlearning.show_optimal()