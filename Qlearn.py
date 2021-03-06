# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:20:16 2018

@author: Flo Wolf
"""
import numpy as np

alp = 0.7
gam = 0.9

grid = np.zeros([10,10],dtype=int)

grid[:,0]= 1
grid[:,-1]=1
grid[0,:]= 1
grid[-1,:]=1
grid[8,8]=3
grid[6,5]=2
grid[7,2:5]=1
grid[3:6,6]=1
grid[2,3:7]=1
print(grid)


q_grid = np.zeros([10,10])
r_grid = -1*np.ones([10,10])
r_grid[8,8]=10
r_grid[6,5]=-20
print(r_grid)

def get_pos(start_pos,move):
    if move == 1:
        new_pos = [start_pos[0],start_pos[1]+1]
    if move == 2:
        new_pos = [start_pos[0],start_pos[1]-1]
    if move == 3:
        new_pos = [start_pos[0]-1,start_pos[1]]
    if move == 4:
        new_pos = [start_pos[0]+1,start_pos[1]]
    if grid[new_pos[0],new_pos[1]]==1:
        new_pos = start_pos
    return new_pos
		



def q_cal(pos):
    opt = []
    for move in [1,2,3,4]:
        new_pos = get_pos(pos,move)
        # if it still doesnt work put this BACK and replace with r
        q_new = r_grid[new_pos[0],new_pos[1]]+gam*q_grid[new_pos[0],new_pos[1]]
        opt.append(q_new)
        max_move = move
    q_max = q_new
    for i in [0,1,2,3]:
        q_holder = opt[i]
        if q_holder >= q_max:
            max_move = i+1
            q_max = q_holder
    new_pos = get_pos(pos,max_move)
    if new_pos == pos:
        q_grid[pos[0],pos[1]]-=1
    else:
        # BACK
        q_now = q_grid[pos[0],pos[1]] + alp*(q_max-q_grid[pos[0],pos[1]])
        q_grid[pos[0],pos[1]] = q_now
    return max_move
    
# Function to play the game

def play(pos):
    stopper=0
    move_list=[]
    pos_list=[pos]
    counter = 0
    while stopper == 0:

        counter += 1
        next_move = q_cal(pos)
        move_list.append(next_move)
        pos = get_pos(pos,next_move)
        pos_list.append(pos)
        if grid[pos[0],pos[1]] == 3 or grid[pos[0],pos[1]] == 2:
            stopper = 1
        
    return counter, move_list, pos_list
        
for i in range(10000):
    c = play([1,1])
    print(c[2])
    print_grid = q_grid
    for i in range(10):
        for j in range(10):
            print_grid[i,j]=round(q_grid[i,j],2)
    print(print_grid)