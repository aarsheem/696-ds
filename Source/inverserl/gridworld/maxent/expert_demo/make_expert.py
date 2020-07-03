import sys
sys.path.append("../")
import numpy as np
from advgridworld import AdvGridworld
import os

#'up','right','down','left','upri','dori','dole','uple'
#actions: [0, 1, 2, 3, 4, 5, 6, 7]

human_policy = [1,1,5,2,3,6,2,\
                2,3,3,2,7,5,2,\
                2,1,3,5,1,1,6,\
                5,1,5,1,1,5,2,\
                2,5,3,2,5,5,2,\
                2,2,5,5,5,5,2,\
                2,1,1,1,1,1,0]


expert = []
for _ in range(10000):
    path = os.getcwd()
    env = AdvGridworld()
    os.chdir(path)
    ep = []
    i = 0
    s = 0
    while i<100:
        if np.random.uniform() > 0.75:
            a = np.random.randint(0,7)
        else:
            a = human_policy[s]
        state,reward,done = env.step(a)
        a=human_policy[s]
        print(state)
        s = state[1]*(env.getBoardDim()[0]+1)+state[0]
        ep.append([s, a, reward])
        if done:
            break
        i+=1
    # print(i)
    ep = np.array(ep)
    expert.append(ep)
print(expert)
expert_np = np.array(expert)
np.save("expert_demo", expert_np)
print(expert_np.shape)
